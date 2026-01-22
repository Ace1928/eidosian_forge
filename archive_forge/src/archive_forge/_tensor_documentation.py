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

        Creates a DLpack `capsule https://data-apis.org/array-api/latest/design_topics/data_interchange.html#data-interchange`_
        of the current tensor to be exported to other libraries.

        This function will be called from the `from_dlpack` method
        of the library that will consume the capsule. `from_dlpack` passes the current
        stream to this method as part of the specification.

        Args:
            stream (integer or None): An optional Python integer representing a
            pointer to a CUDA stream. The current stream is synchronized with
            this stream before the capsule is created, and since the capsule
            shares its storage with the tensor this make it safe to access from
            both streams.  If None or -1 is passed then no synchronization is performed.
            If 1 (on CUDA) or 0 (on ROCM) then the default stream is used for
            synchronization.
        