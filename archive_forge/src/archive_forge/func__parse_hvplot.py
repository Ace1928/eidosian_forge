from typing import Any, Dict, Tuple
import panel as _pn
from . import hvPlotTabular, post_patch
from .util import _fugue_ipython
@parse_outputter.candidate(namespace_candidate(name, lambda x: isinstance(x, str)))
def _parse_hvplot(obj: Tuple[str, str]) -> Outputter:
    return _Visualize(obj[1])