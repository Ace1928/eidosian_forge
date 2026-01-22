import io
from collections import defaultdict
from itertools import filterfalse
from typing import Dict, List, Tuple, Mapping, TypeVar
from .. import _reqs
from ..extern.jaraco.text import yield_lines
from ..extern.packaging.requirements import Requirement
def _convert_extras_requirements(extras_require: _StrOrIter) -> Mapping[str, _Ordered[Requirement]]:
    """
    Convert requirements in `extras_require` of the form
    `"extra": ["barbazquux; {marker}"]` to
    `"extra:{marker}": ["barbazquux"]`.
    """
    output: Mapping[str, _Ordered[Requirement]] = defaultdict(dict)
    for section, v in extras_require.items():
        output[section]
        for r in _reqs.parse(v):
            output[section + _suffix_for(r)].setdefault(r)
    return output