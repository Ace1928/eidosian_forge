import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
from torch.overrides import (
from torch.utils.dlpack import DLDeviceType
def to_sparse_coo(self):
    """Convert a tensor to :ref:`coordinate format <sparse-coo-docs>`.

        Examples::

             >>> dense = torch.randn(5, 5)
             >>> sparse = dense.to_sparse_coo()
             >>> sparse._nnz()
             25

        """
    return self.to_sparse()