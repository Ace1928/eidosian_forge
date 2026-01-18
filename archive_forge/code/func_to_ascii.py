import codecs
import re
from typing import Callable
def to_ascii(obj: str) -> str:

    def mapping(obj: str) -> str:
        if REGEX_NON_ASCII.search(obj):
            return 'xn--' + encode(obj)
        return obj
    return map_domain(obj, mapping)