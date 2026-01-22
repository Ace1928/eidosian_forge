import re
from .core import BlockState
from .util import (
def _compile_list_item_pattern(bullet, leading_width):
    if leading_width > 3:
        leading_width = 3
    return '^(?P<listitem_1> {0,' + str(leading_width) + '})(?P<listitem_2>' + bullet + ')(?P<listitem_3>[ \\t]*|[ \\t][^\\n]+)$'