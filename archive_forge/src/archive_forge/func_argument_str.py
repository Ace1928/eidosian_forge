from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def argument_str(self, *, method: bool=False, symint: bool=True) -> str:
    type_str = argument_type_str(self.type, symint=symint).replace('const ', '').replace(' &', '')
    name = self.name
    if name == 'self' and type_str in ['Tensor', 'Number'] and (not method):
        name = 'input'
    if self.default is not None:
        default = {'nullptr': 'None', 'c10::nullopt': 'None', '{}': 'None'}.get(self.default, self.default)
        return f'{type_str} {name}={default}'
    else:
        return f'{type_str} {name}'