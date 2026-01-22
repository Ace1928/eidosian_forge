from ...Qt import QtCore, QtWidgets, QtGui
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
class ParameterControlledButton(QtWidgets.QPushButton):
    settableAttributes = {'title', 'tip', 'icon', 'shortcut', 'enabled', 'visible'}

    def __init__(self, parameter=None, parent=None):
        super().__init__(parent)
        if not parameter:
            return
        parameter.sigNameChanged.connect(self.onNameChange)
        parameter.sigOptionsChanged.connect(self.updateOpts)
        self.clicked.connect(parameter.activate)
        self.updateOpts(parameter, parameter.opts)

    def updateOpts(self, param, opts):
        nameMap = dict(title='text', tip='toolTip')
        opts = opts.copy()
        if 'name' in opts:
            opts.setdefault('title', opts['name'])
        if 'title' in opts and opts['title'] is None:
            opts['title'] = param.title()
        if 'icon' in opts:
            opts['icon'] = QtGui.QIcon(opts['icon'])
        for attr in self.settableAttributes.intersection(opts):
            buttonAttr = nameMap.get(attr, attr)
            capitalized = buttonAttr[0].upper() + buttonAttr[1:]
            setter = getattr(self, f'set{capitalized}')
            setter(opts[attr])

    def onNameChange(self, param, name):
        self.updateOpts(param, dict(title=param.title()))