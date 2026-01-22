from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def _cpp_signature(f: NativeFunction, *, method: bool=False) -> CppSignature:
    return CppSignatureGroup.from_native_function(f, method=method).signature