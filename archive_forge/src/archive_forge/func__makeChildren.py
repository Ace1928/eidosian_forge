import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
def _makeChildren(self, boundPen=None):
    cs = QtCore.Qt.PenCapStyle
    js = QtCore.Qt.PenJoinStyle
    ps = QtCore.Qt.PenStyle
    param = Parameter.create(name='Params', type='group', children=[dict(name='color', type='color', value='k'), dict(name='width', value=1, type='int', limits=[0, None]), QtEnumParameter(ps, name='style', value='SolidLine'), QtEnumParameter(cs, name='capStyle'), QtEnumParameter(js, name='joinStyle'), dict(name='cosmetic', type='bool', value=True)])
    optsPen = boundPen or fn.mkPen()
    for p in param:
        name = p.name()
        if p.type() == 'bool':
            attrName = f'is{name.title()}'
        else:
            attrName = name
        default = getattr(optsPen, attrName)()
        replace = '\\1 \\2'
        name = re.sub('(\\w)([A-Z])', replace, name)
        name = name.title().strip()
        p.setOpts(title=name, default=default)
    if boundPen is not None:
        self.updateFromPen(param, boundPen)
        for p in param:
            setName = f'set{cap_first(p.name())}'
            setattr(boundPen, setName, p.setValue)
            newSetter = self.penPropertySetter
            if p.type() != 'color':
                p.sigValueChanging.connect(newSetter)
            try:
                p.sigValueChanged.disconnect(p._emitValueChanged)
            except RuntimeError:
                assert p.receivers(QtCore.SIGNAL('sigValueChanged(PyObject,PyObject)')) == 1
                p.sigValueChanged.disconnect()
            p.sigValueChanged.connect(newSetter)
    return param