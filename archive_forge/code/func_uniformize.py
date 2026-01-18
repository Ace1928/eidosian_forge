import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List
import numpy as np
from numpy import inf
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo
def uniformize(usv):
    S = usv[1]
    k = S.shape[-1]
    U = usv[0][..., :k]
    Vh = usv[2] if is_linalg_svd else usv[2].mH
    Vh = Vh[..., :k, :]
    return (U, S, Vh)