from _pydevd_bundle.pydevd_constants import IS_PY311_OR_GREATER
import dis
from types import CodeType
from collections import namedtuple
def _is_inside(item_pos: _Pos, container_pos: _Pos):
    if item_pos.lineno < container_pos.lineno or item_pos.endlineno > container_pos.endlineno:
        return False
    if item_pos.lineno == container_pos.lineno:
        if item_pos.startcol < container_pos.startcol:
            return False
    if item_pos.endlineno == container_pos.endlineno:
        if item_pos.endcol > container_pos.endcol:
            return False
    return True