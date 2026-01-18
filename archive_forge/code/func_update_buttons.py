from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def update_buttons(self):
    valid = True
    for field in self.float_fields:
        if not is_edit_valid(field):
            valid = False
    for btn_type in ['Ok', 'Apply']:
        btn = self.bbox.button(getattr(QtWidgets.QDialogButtonBox.StandardButton, btn_type))
        if btn is not None:
            btn.setEnabled(valid)