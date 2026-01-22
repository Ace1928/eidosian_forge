import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
def _real_re_compile(self, *args, **kwargs):
    self._actions.append(('_real_re_compile', args, kwargs))
    return super()._real_re_compile(*args, **kwargs)