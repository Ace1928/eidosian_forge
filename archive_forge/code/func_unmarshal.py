import sys
import json
from .symbols import *
from .symbols import Symbol
def unmarshal(self, d):
    if isinstance(d, dict):
        return {self._unescape(k): self.unmarshal(v) for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        return type(d)((self.unmarshal(x) for x in d))
    else:
        return self._unescape(d)