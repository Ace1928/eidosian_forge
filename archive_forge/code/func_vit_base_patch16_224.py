import math
import re
from collections import OrderedDict
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.helpers import named_apply
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth
from flash_attn.layers.patch_embed import PatchEmbed
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    assert not pretrained
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model