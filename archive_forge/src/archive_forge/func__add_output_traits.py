from ... import logging
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list
from ...utils.functions import getsource, create_function_from_source
def _add_output_traits(self, base):
    undefined_traits = {}
    for key in self._output_names:
        base.add_trait(key, traits.Any)
        undefined_traits[key] = Undefined
    base.trait_set(trait_change_notify=False, **undefined_traits)
    return base