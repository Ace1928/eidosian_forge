import io
from collections import defaultdict
from itertools import filterfalse
from typing import Dict, List, Tuple, Mapping, TypeVar
from .. import _reqs
from ..extern.jaraco.text import yield_lines
from ..extern.packaging.requirements import Requirement
def _suffix_for(req):
    """Return the 'extras_require' suffix for a given requirement."""
    return ':' + str(req.marker) if req.marker else ''