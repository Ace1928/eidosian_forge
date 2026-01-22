from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class VLM(nn.Module):
    """
    Vision-Language Model for Multimodal embeddings.
    """

    def __init__(self, config: Dict, tokenizer_path: PathLike):
        """
        :param config: Model config
        """
        super().__init__()
        self._embedding_dim = config['text_encoder']['embedding_dim']
        self.text_encoder = TextEncoder(**config['text_encoder'])
        self.image_encoder = VisualEncoder(**config['image_encoder'])

    def encode_image(self, images: Tensor, return_features: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Passes the pre-processed images through `image_encoder` to produce images features (optional) and embeddings.

        :param images: Preprocessed image
        :param return_features: Whether to return images features or return only embeddings
        """
        features = self.image_encoder.forward_features(images)
        embeddings = self.image_encoder.forward_embedding(features)
        if return_features:
            return (features, embeddings)
        return embeddings

    def encode_text(self, texts: Dict[str, Tensor], return_features: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Passes the pre-processed texts through `text_encoder` to produce texts features (optional) and embeddings.

        :param texts: Dictionary with tokenized texts and attention masks
        :param return_features: Whether to return texts features or return only embeddings
        """
        features = self.text_encoder.forward_features(texts['input_ids'], texts['attention_mask'])
        embeddings = self.text_encoder.forward_embedding(features, texts['attention_mask'])
        if return_features:
            return (features, embeddings)
        return embeddings

    def encode_multimodal(self, image: Optional[Tensor]=None, text: Optional[Dict]=None, image_features: Optional[Tensor]=None, text_features: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, return_scores: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Passes preprocessed texts (or precomputed texts features) and
            preprocessed images (or precomputed images features) through multimodal encoded to produce multimodal joint embeddings.

        :param image: Preprocessed images
        :param text: Preprocessed texts
        :param image_features: Precomputed images features
        :param text_features: Precomputed text features
        :param attention_mask: Attention masks, not required if pass `text` instead of text_features
        """
        assert image is not None or image_features is not None, 'Either `image` or `image_features` should be non None'
        assert text is not None or text_features is not None, 'Either `text_data` or `text_features` should be non None'
        if text_features is not None:
            assert attention_mask is not None, 'if `text_features` is not None, then you should pass `attention_mask`'
        if image_features is None:
            image_features = self.image_encoder.forward_features(image)
        if text_features is None:
            text_features = self.text_encoder.forward_features(text['input_ids'], text['attention_mask'])
        embeddings = self.text_encoder.forward_multimodal(text_features, attention_mask if attention_mask is not None else text['attention_mask'], image_features)
        if return_scores:
            return (self.get_matching_scores(embeddings), embeddings)
        return embeddings

    def get_matching_scores(self, embeddings: Tensor) -> Tensor:
        """Computes the probability that there is a match between images and texts based on their multimodal embeddings

        :param embeddings: multimodal joint embeddings
        """
        return self.text_encoder.forward_matching(embeddings)

    def forward(self, images: Tensor, texts: Dict[str, Tensor]) -> Union[Tensor, Tensor]:
        """Inference forward method

        :param images: Preprocessed images
        :param texts: Preprocessed texts
        :return: embeddings for images and texts
        """
        _, image_embeddings = self.image_encoder(images)
        _, text_embeddings = self.text_encoder(texts)
        return (image_embeddings, text_embeddings)

    @property
    def text_features_dim(self) -> int:
        """Dimensionality of the text encoder features."""
        return self.text_encoder.dim

    @property
    def image_features_dim(self) -> int:
        """Dimensionality of the image encoder features."""
        return self.image_encoder.dim

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of shared space embedding."""
        return self._embedding_dim

    @property
    def multimodal_embedding_dim(self) -> int:
        """Dimensionality of multimodal joint embedding."""
        return self.text_encoder.dim