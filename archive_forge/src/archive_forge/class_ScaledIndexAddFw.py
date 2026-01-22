from typing import Optional, Sequence
import torch
from xformers.ops._triton import (
from .common import BaseOperator, register_operator
@register_operator
class ScaledIndexAddFw(BaseOperator):
    OPERATOR = scaled_index_add_fwd
    OPERATOR_CATEGORY = 'indexing'
    NAME = 'scaled_index_addF'