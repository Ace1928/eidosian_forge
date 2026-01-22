from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention
from transformers.models.bloom.modeling_bloom import BloomAttention
from transformers.models.codegen.modeling_codegen import CodeGenAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention
from transformers.models.marian.modeling_marian import MarianAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.pegasus.modeling_pegasus import PegasusAttention
from transformers.models.t5.modeling_t5 import T5Attention
from ...utils.import_utils import check_if_transformers_greater
from .attention import (
from .base import BetterTransformerBaseLayer
class BlenderbotAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BlenderbotAttention, nn.Module):

    def __init__(self, layer: 'nn.Module', config: 'PretrainedConfig'):
        super().__init__(config)
        bart_bettertransformer_init(self, layer, config)

    def forward(self, *args, **kwargs):
        return bart_forward(self, *args, **kwargs)