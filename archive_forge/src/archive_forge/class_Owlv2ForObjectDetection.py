import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
class Owlv2ForObjectDetection(Owlv2PreTrainedModel):
    config_class = Owlv2Config

    def __init__(self, config: Owlv2Config):
        super().__init__(config)
        self.owlv2 = Owlv2Model(config)
        self.class_head = Owlv2ClassPredictionHead(config)
        self.box_head = Owlv2BoxPredictionHead(config)
        self.objectness_head = Owlv2BoxPredictionHead(config, out_dim=1)
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        self.sigmoid = nn.Sigmoid()

    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        if not feature_map.ndim == 4:
            raise ValueError('Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]')
        device = feature_map.device
        num_patches = feature_map.shape[1]
        box_coordinates = np.stack(np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)
        box_coordinates = box_coordinates.reshape(box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2])
        box_coordinates = torch.from_numpy(box_coordinates).to(device)
        return box_coordinates

    def objectness_predictor(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Predicts the probability that each image feature token is an object.

        Args:
            image_features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`)):
                Features extracted from the image.
        Returns:
            Objectness scores.
        """
        image_features = image_features.detach()
        objectness_logits = self.objectness_head(image_features)
        objectness_logits = objectness_logits[..., 0]
        return objectness_logits

    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)
        box_coord_bias = torch.log(box_coordinates + 0.0001) - torch.log1p(-box_coordinates + 0.0001)
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 0.0001) - torch.log1p(-box_size + 0.0001)
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def box_predictor(self, image_feats: torch.FloatTensor, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        pred_boxes = self.box_head(image_feats)
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(self, image_feats: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor]=None, query_mask: Optional[torch.Tensor]=None) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        pred_logits, image_class_embeds = self.class_head(image_feats, query_embeds, query_mask)
        return (pred_logits, image_class_embeds)

    def image_text_embedder(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: torch.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Tuple[torch.FloatTensor]:
        outputs = self.owlv2(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        last_hidden_state = outputs.vision_model_output[0]
        image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)
        new_size = (image_embeds.shape[0], int(np.sqrt(image_embeds.shape[1])), int(np.sqrt(image_embeds.shape[1])), image_embeds.shape[-1])
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs[-4]
        return (text_embeds, image_embeds, outputs)

    def image_embedder(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Tuple[torch.FloatTensor]:
        vision_outputs = self.owlv2.vision_model(pixel_values=pixel_values, return_dict=True)
        last_hidden_state = vision_outputs[0]
        image_embeds = self.owlv2.vision_model.post_layernorm(last_hidden_state)
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)
        new_size = (image_embeds.shape[0], int(np.sqrt(image_embeds.shape[1])), int(np.sqrt(image_embeds.shape[1])), image_embeds.shape[-1])
        image_embeds = image_embeds.reshape(new_size)
        return (image_embeds, vision_outputs)

    def embed_image_query(self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor) -> torch.FloatTensor:
        _, class_embeds = self.class_predictor(query_image_features)
        pred_boxes = self.box_predictor(query_image_features, query_feature_map)
        pred_boxes_as_corners = center_to_corners_format(pred_boxes)
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device
        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)
            iou_threshold = torch.max(ious) * 0.8
            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum('d,id->i', mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)
        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = (None, None)
        return (query_embeds, box_indices, pred_boxes)

    @add_start_docstrings_to_model_forward(OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2ImageGuidedObjectDetectionOutput, config_class=Owlv2Config)
    def image_guided_detection(self, pixel_values: torch.FloatTensor, query_pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Owlv2ImageGuidedObjectDetectionOutput:
        """
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> import numpy as np
        >>> from transformers import AutoProcessor, Owlv2ForObjectDetection
        >>> from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
        >>> query_image = Image.open(requests.get(query_url, stream=True).raw)
        >>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model.image_guided_detection(**inputs)

        >>> # Note: boxes need to be visualized on the padded, unnormalized image
        >>> # hence we'll set the target image sizes (height, width) based on that

        >>> def get_preprocessed_image(pixel_values):
        ...     pixel_values = pixel_values.squeeze().numpy()
        ...     unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        ...     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        ...     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        ...     unnormalized_image = Image.fromarray(unnormalized_image)
        ...     return unnormalized_image

        >>> unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        >>> target_sizes = torch.Tensor([unnormalized_image.size[::-1]])

        >>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> results = processor.post_process_image_guided_detection(
        ...     outputs=outputs, threshold=0.9, nms_threshold=0.3, target_sizes=target_sizes
        ... )
        >>> i = 0  # Retrieve predictions for the first image
        >>> boxes, scores = results[i]["boxes"], results[i]["scores"]
        >>> for box, score in zip(boxes, scores):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
        Detected similar object with confidence 0.938 at location [490.96, 109.89, 821.09, 536.11]
        Detected similar object with confidence 0.959 at location [8.67, 721.29, 928.68, 732.78]
        Detected similar object with confidence 0.902 at location [4.27, 720.02, 941.45, 761.59]
        Detected similar object with confidence 0.985 at location [265.46, -58.9, 1009.04, 365.66]
        Detected similar object with confidence 1.0 at location [9.79, 28.69, 937.31, 941.64]
        Detected similar object with confidence 0.998 at location [869.97, 58.28, 923.23, 978.1]
        Detected similar object with confidence 0.985 at location [309.23, 21.07, 371.61, 932.02]
        Detected similar object with confidence 0.947 at location [27.93, 859.45, 969.75, 915.44]
        Detected similar object with confidence 0.996 at location [785.82, 41.38, 880.26, 966.37]
        Detected similar object with confidence 0.998 at location [5.08, 721.17, 925.93, 998.41]
        Detected similar object with confidence 0.969 at location [6.7, 898.1, 921.75, 949.51]
        Detected similar object with confidence 0.966 at location [47.16, 927.29, 981.99, 942.14]
        Detected similar object with confidence 0.924 at location [46.4, 936.13, 953.02, 950.78]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        query_feature_map = self.image_embedder(pixel_values=query_pixel_values)[0]
        feature_map, vision_outputs = self.image_embedder(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
        query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        query_embeds, best_box_indices, query_pred_boxes = self.embed_image_query(query_image_feats, query_feature_map)
        pred_logits, class_embeds = self.class_predictor(image_feats=image_feats, query_embeds=query_embeds)
        target_pred_boxes = self.box_predictor(image_feats, feature_map)
        if not return_dict:
            output = (feature_map, query_feature_map, target_pred_boxes, query_pred_boxes, pred_logits, class_embeds, vision_outputs.to_tuple())
            output = tuple((x for x in output if x is not None))
            return output
        return Owlv2ImageGuidedObjectDetectionOutput(image_embeds=feature_map, query_image_embeds=query_feature_map, target_pred_boxes=target_pred_boxes, query_pred_boxes=query_pred_boxes, logits=pred_logits, class_embeds=class_embeds, text_model_output=None, vision_model_output=vision_outputs)

    @add_start_docstrings_to_model_forward(OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2ObjectDetectionOutput, config_class=Owlv2Config)
    def forward(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Owlv2ObjectDetectionOutput:
        """
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import numpy as np
        >>> import torch
        >>> from transformers import AutoProcessor, Owlv2ForObjectDetection
        >>> from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = [["a photo of a cat", "a photo of a dog"]]
        >>> inputs = processor(text=texts, images=image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Note: boxes need to be visualized on the padded, unnormalized image
        >>> # hence we'll set the target image sizes (height, width) based on that

        >>> def get_preprocessed_image(pixel_values):
        ...     pixel_values = pixel_values.squeeze().numpy()
        ...     unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        ...     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        ...     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        ...     unnormalized_image = Image.fromarray(unnormalized_image)
        ...     return unnormalized_image

        >>> unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        >>> target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        >>> # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        >>> results = processor.post_process_object_detection(
        ...     outputs=outputs, threshold=0.2, target_sizes=target_sizes
        ... )

        >>> i = 0  # Retrieve predictions for the first image for the corresponding text queries
        >>> text = texts[i]
        >>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        >>> for box, score, label in zip(boxes, scores, labels):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        Detected a photo of a cat with confidence 0.614 at location [512.5, 35.08, 963.48, 557.02]
        Detected a photo of a cat with confidence 0.665 at location [10.13, 77.94, 489.93, 709.69]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        query_embeds, feature_map, outputs = self.image_text_embedder(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_outputs = outputs.text_model_output
        vision_outputs = outputs.vision_model_output
        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0
        pred_logits, class_embeds = self.class_predictor(image_feats, query_embeds, query_mask)
        objectness_logits = self.objectness_predictor(image_feats)
        pred_boxes = self.box_predictor(image_feats, feature_map)
        if not return_dict:
            output = (pred_logits, objectness_logits, pred_boxes, query_embeds, feature_map, class_embeds, text_outputs.to_tuple(), vision_outputs.to_tuple())
            output = tuple((x for x in output if x is not None))
            return output
        return Owlv2ObjectDetectionOutput(image_embeds=feature_map, text_embeds=query_embeds, pred_boxes=pred_boxes, logits=pred_logits, objectness_logits=objectness_logits, class_embeds=class_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)