from typing import Dict, List, Optional, Sequence, Tuple
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import FileManager
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
Generate an additional lambda function to recover views in backward when as_strided is not supported.
    See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details.
    