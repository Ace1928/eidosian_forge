from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def _to_list_of_ints(s):
    s = s.replace(b',', b' ')
    return [_to_int(val) for val in s.split()]