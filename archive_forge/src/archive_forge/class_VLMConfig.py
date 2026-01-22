from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
from transformers import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (AutoModel,
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from uform.torch_models import VisualEncoder
class VLMConfig(PretrainedConfig):
    model_type = 'vlm'

    def __init__(self, text_decoder_name_or_path: str='', tokenizer_name_or_path: str='', image_size: int=224, image_encoder_hidden_size: int=768, image_encoder_patch_size: int=16, image_encoder_num_layers: int=12, image_encoder_num_heads: int=12, image_encoder_embedding_dim: int=256, image_encoder_pooling: str='cls', image_pooler_num_attn_heads: int=16, image_pooler_intermediate_size: int=5504, image_pooler_num_latents: int=196, image_token_id: int=32002, initializer_range: float=0.02, use_cache: bool=True, center_crop: bool=True, **kwargs):
        self.text_decoder_name_or_path = text_decoder_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.image_size = image_size
        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.image_encoder_patch_size = image_encoder_patch_size
        self.image_encoder_num_layers = image_encoder_num_layers
        self.image_encoder_num_heads = image_encoder_num_heads
        self.image_encoder_embedding_dim = image_encoder_embedding_dim
        self.image_encoder_pooling = image_encoder_pooling
        self.image_pooler_num_attn_heads = image_pooler_num_attn_heads
        self.image_pooler_intermediate_size = image_pooler_intermediate_size
        self.image_pooler_num_latents = image_pooler_num_latents
        self.image_token_id = image_token_id
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.center_crop = center_crop
        super().__init__(**kwargs)