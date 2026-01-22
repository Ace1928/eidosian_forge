import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from functorch.compile import compiled_function, min_cut_rematerialization_partition, nop
from .make_fx import randomize
import re
class assert_raises_regex:

    def __init__(self, exception_cls, regex):
        self.exception_cls = exception_cls
        self.regex = regex

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type == self.exception_cls:
            msg = str(exc_val)
            if not re.search(self.regex, msg):
                raise AssertionError(f'Expected exception to match regex. regex: {self.regex}, exception: {msg}')
            return True
        if exc_type is not None:
            raise AssertionError(f'Expected {self.exception_cls} to be raised, instead got exception {exc_type}')
        raise AssertionError('Expected exception to be raised but none was')