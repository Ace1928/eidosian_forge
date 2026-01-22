import torch
import re
import unittest
from subprocess import CalledProcessError
from torch._inductor.codecache import CppCodeCache
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import (
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase
def _check_has_dynamic_shape(self: TestCase, code):
    for_loop_found = False
    has_dynamic = False
    lines = code.split('\n')
    for line in lines:
        if 'for(' in line:
            for_loop_found = True
            if re.search(';.*ks.*;', line) is not None:
                has_dynamic = True
                break
    self.assertTrue(has_dynamic, msg=f'Failed to find dynamic for loop variable\n{code}')
    self.assertTrue(for_loop_found, f'Failed to find for loop\n{code}')