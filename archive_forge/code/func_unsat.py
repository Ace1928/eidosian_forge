from typing import Dict, List, NoReturn, Sequence, Union
from torchgen.api.types import (
def unsat(goal: NamedCType) -> NoReturn:
    ctx_desc = '\n'.join((f'  {t.cpp_type()} {t.name}; // {e}' for t, e in ctx.items()))
    raise UnsatError(f'\nFailed to synthesize the expression "{goal.cpp_type()} {goal.name}".\nWhen I failed, the following bindings were available in the context:\n\n{ctx_desc}\n\nThis probably means there is a missing rule in the rules of torchgen.api.translate.\nCheck this module for more information.\n')