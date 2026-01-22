from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
from torchgen.api.types import Binding, CType, NamedCType
from torchgen.model import (
def _gen_code_optional_type(self, arg_name: str, out_name: str, t: OptionalType, ctype: CType) -> Tuple[List[str], List[str]]:
    in_name = f'{arg_name}_opt_in'
    res_name, base_type, res_code, decl = self.argumenttype_evalue_convert(t.elem, in_name)
    return (f'\n    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toOptional<{base_type.cpp_type(strip_ref=True)}>();\n            '.split('\n'), decl)