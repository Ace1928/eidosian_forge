import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_unispeech_sat import UniSpeechSatConfig
@add_start_docstrings('UniSpeechSat Model with a quantizer and `VQ` head on top.', UNISPEECH_SAT_START_DOCSTRING)
class UniSpeechSatForPreTraining(UniSpeechSatPreTrainedModel):

    def __init__(self, config: UniSpeechSatConfig):
        super().__init__(config)
        self.unispeech_sat = UniSpeechSatModel(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)
        self.quantizer = UniSpeechSatGumbelVectorQuantizer(config)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.dropout = nn.Dropout(config.final_dropout)
        self.speaker_proj = nn.Linear(config.hidden_size, config.codevector_dim)
        self.label_embeddings_concat = nn.Parameter(torch.FloatTensor(config.num_clusters, config.codevector_dim))
        self.label_embeddings_concat.data.zero_()
        self.layer_norm_for_extract = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if self.config.do_stable_layer_norm:
            self.layer_norm_for_extract.requires_grad = False
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(target_features: torch.FloatTensor, negative_features: torch.FloatTensor, predicted_features: torch.FloatTensor, temperature: int=1):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1)
        logits = logits.type_as(target_features)
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UniSpeechSatForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, UniSpeechSatForPreTrainingOutput]:
        """
        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, UniSpeechSatForPreTraining
        >>> from transformers.models.unispeech_sat.modeling_unispeech_sat import _compute_mask_indices

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base")
        >>> model = UniSpeechSatForPreTraining.from_pretrained("microsoft/unispeech-sat-base")
        >>> # TODO: Add full pretraining example
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.unispeech_sat(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        transformer_features = outputs[0]
        extract_features = self.dropout_features(outputs[1])
        logits = extract_features
        loss = quantized_features = codevector_perplexity = None
        if not return_dict:
            if loss is not None:
                return (loss, logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
        return UniSpeechSatForPreTrainingOutput(loss=loss, logits=logits, projected_states=transformer_features, projected_quantized_states=quantized_features, codevector_perplexity=codevector_perplexity, hidden_states=outputs.hidden_states, attentions=outputs.attentions)