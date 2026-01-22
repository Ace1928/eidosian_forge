import torch
from ..modules import Module
from . import comm
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, TypeVar, Union, cast
from torch._utils import _get_device_index
from collections import OrderedDict
def _replicatable_module(module: Module, memo: Optional[Set[Module]]=None) -> bool:

    def descendant_modules(module: Module) -> Iterator[Module]:
        gen = module.modules()
        next(gen)
        return gen
    if not _is_jit_enabled():
        return True
    if memo is None:
        memo = set()
    memo.add(module)
    if _is_script_module(module):
        memo.update(descendant_modules(module))
        return all((_is_script_module(descendant) for descendant in descendant_modules(module)))
    for child in module.children():
        if child in memo:
            continue
        if not _replicatable_module(child, memo):
            return False
    return True