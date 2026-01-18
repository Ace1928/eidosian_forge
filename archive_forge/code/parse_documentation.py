import re
from functools import partial
from typing import Any, Callable, Dict, Tuple
from curtsies.formatstring import fmtstr, FmtStr
from curtsies.termformatconstants import (
from ..config import COLOR_LETTERS
from ..lazyre import LazyReCompile
Returns a FmtStr object from a bpython-formatted colored string