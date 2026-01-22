import copy
import re
import types
from .ucre import build_re
def _create_validator(self, regex):

    def func(text, pos):
        tail = text[pos:]
        if isinstance(regex, str):
            founds = re.search(regex, tail, flags=re.IGNORECASE)
        else:
            founds = re.search(regex, tail)
        if founds:
            return len(founds.group(0))
        return 0
    return func