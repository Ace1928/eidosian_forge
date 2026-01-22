import sys
from contextlib import contextmanager
import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule
class MkldnnModule(PropModule):

    def __init__(self, m, name):
        super().__init__(m, name)
    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)