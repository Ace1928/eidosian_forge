from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
import atexit
import enum
import mmap
import os
import sys
import tempfile
from .. import Qt
from .. import CONFIG_OPTIONS
from .. import multiprocess as mp
from .GraphicsView import GraphicsView
class WheelEvent(QtGui.QWheelEvent):

    @staticmethod
    def get_state(obj, picklable=False):
        lpos = obj.position() if hasattr(obj, 'position') else obj.posF()
        gpos = obj.globalPosition() if hasattr(obj, 'globalPosition') else obj.globalPosF()
        pixdel, angdel, btns = (obj.pixelDelta(), obj.angleDelta(), obj.buttons())
        mods, phase, inverted = (obj.modifiers(), obj.phase(), obj.inverted())
        if picklable:
            btns, mods, phase = serialize_mouse_enum(btns, mods, phase)
        return (lpos, gpos, pixdel, angdel, btns, mods, phase, inverted)

    def __init__(self, rhs):
        items = list(self.get_state(rhs))
        items[1] = items[0]
        super().__init__(*items)

    def __getstate__(self):
        return self.get_state(self, picklable=True)

    def __setstate__(self, state):
        pos, gpos, pixdel, angdel, btns, mods, phase, inverted = state
        if not isinstance(btns, enum.Enum):
            btns = QtCore.Qt.MouseButtons(btns)
        if not isinstance(mods, enum.Enum):
            mods = QtCore.Qt.KeyboardModifiers(mods)
        phase = QtCore.Qt.ScrollPhase(phase)
        super().__init__(pos, gpos, pixdel, angdel, btns, mods, phase, inverted)