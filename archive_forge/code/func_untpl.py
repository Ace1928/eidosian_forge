import copy
import re
import types
from .ucre import build_re
def untpl(tpl):
    return tpl.replace('%TLDS%', self.re['src_tlds'])