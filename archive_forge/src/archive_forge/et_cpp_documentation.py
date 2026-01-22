from typing import List, Optional, Sequence, Set, Union
from torchgen import local
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
from .types import (

This file describes the translation of JIT schema to the public C++ API, which is what people use when they call
functions like at::add. It also serves as a native function API, which is the signature of kernels,
since in Executorch CppSignature is the same as NativeSignature.

Difference between this file and torchgen.api.cpp.py:

  - Executorch doesn't support TensorOptions, however in this file we still keep the logic here to be compatible with
    torchgen.api.cpp, so that we can do stuff like ATen mode (running ATen kernels in Executorch).

  - Executorch doesn't support Dimname.

  - Executorch runtime doesn't support SymInt, will treat it as int.
