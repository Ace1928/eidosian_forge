from .. import functions as fn
from .parameterTypes import GroupParameter
from .SystemSolver import SystemSolver
def updateParamState(self, param, state):
    if state == 'autoSet':
        bg = fn.mkBrush((200, 255, 200, 255))
        bold = False
        readonly = True
    elif state == 'autoUnset':
        bg = fn.mkBrush(None)
        bold = False
        readonly = False
    elif state == 'fixed':
        bg = fn.mkBrush('y')
        bold = True
        readonly = False
    else:
        raise ValueError("'state' must be one of 'autoSet', 'autoUnset', or 'fixed'")
    param.setReadonly(readonly)