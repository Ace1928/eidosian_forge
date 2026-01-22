import inspect
import logging
import torch
from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint, uid
import torch._dynamo.config
class Wrap(HigherOrderOperator):

    def __init__(self):
        super().__init__('wrap')

    def __call__(self, func, *args, **kwargs):
        import torch._dynamo
        from torch._dynamo import disable

        @disable
        def wrapper():
            result = func(*args, **kwargs)
            return result
        return wrapper()