import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
def _parse_docs():
    import re
    import textwrap
    doc = regionprops.__doc__ or ''
    matches = re.finditer('\\*\\*(\\w+)\\*\\* \\:.*?\\n(.*?)(?=\\n    [\\*\\S]+)', doc, flags=re.DOTALL)
    prop_doc = {m.group(1): textwrap.dedent(m.group(2)) for m in matches}
    return prop_doc