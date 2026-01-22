from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
def _rebase_path(value, cwd):
    if isinstance(value, list):
        return [_rebase_path(v, cwd) for v in value]
    try:
        value = Path(value)
    except TypeError:
        pass
    else:
        try:
            value = value.relative_to(cwd)
        except ValueError:
            pass
    return value