import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
def wrapper_registrations(used_keys: Set[str]) -> str:
    library_impl_macro_list: List[str] = []
    for key in sorted(used_keys):
        dispatch_key = key
        if key == 'Default':
            dispatch_key = 'Autograd'
        library_impl_macro = f'TORCH_LIBRARY_IMPL(aten, {dispatch_key}, m) ' + '{\n' + '${' + f'wrapper_registrations_{key}' + '}\n}'
        library_impl_macro_list += [library_impl_macro]
    return '\n\n'.join(library_impl_macro_list)