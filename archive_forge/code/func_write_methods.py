import collections
import textwrap
from dataclasses import dataclass, field
from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core import duck_array_ops
from xarray.core.options import OPTIONS
from xarray.core.types import Dims, Self
from xarray.core.utils import contains_only_chunked_or_numpy, module_available
from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Callable
from xarray.core import duck_array_ops
from xarray.core.types import Dims, Self
def write_methods(filepath, generators, preamble):
    with open(filepath, mode='w', encoding='utf-8') as f:
        f.write(preamble)
        for gen in generators:
            for lines in gen.generate_methods():
                for line in lines:
                    f.write(line + '\n')