from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def _constructMetaBtns(self):
    self.metaBtnWidget = QtWidgets.QWidget()
    self.metaBtnLayout = lay = QtWidgets.QHBoxLayout(self.metaBtnWidget)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    self.metaBtns = {}
    lay.addStretch(0)
    for title in ('Clear', 'Select'):
        self.metaBtns[title] = btn = QtWidgets.QPushButton(f'{title} All')
        self.metaBtnLayout.addWidget(btn)
        btn.clicked.connect(getattr(self, f'{title.lower()}AllClicked'))
    self.metaBtns['default'] = self.makeDefaultButton()
    self.metaBtnLayout.addWidget(self.metaBtns['default'])