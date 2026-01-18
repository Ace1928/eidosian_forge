from copy import copy
from ase.gui.i18n import _
import numpy as np
import ase
import ase.data
import ase.gui.ui as ui
from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.cluster.hexagonal import HexagonalClosedPacked, Graphite
from ase.cluster import wulff_construction
from ase.gui.widgets import Element, pybutton
import ase
import ase
from ase.cluster import wulff_construction
def update_new_direction_and_size_stuff(self):
    if self.needs_4index[self.structure.value]:
        n = 4
    else:
        n = 3
    rows = self.new_direction_and_size_rows
    rows.clear()
    self.new_direction = row = ['(']
    for i in range(n):
        if i > 0:
            row.append(',')
        row.append(ui.SpinBox(0, -100, 100, 1))
    row.append('):')
    if self.method.value == 'wulff':
        row.append(ui.SpinBox(1.0, 0.0, 1000.0, 0.1))
    else:
        row.append(ui.SpinBox(5, 1, 100, 1))
    row.append(ui.Button(_('Add'), self.row_add))
    rows.add(row)
    if self.method.value == 'wulff':
        self.size_radio = ui.RadioButtons([_('Number of atoms'), _('Diameter')], ['natoms', 'diameter'], self.update_gui_size)
        self.size_natoms = ui.SpinBox(100, 1, 100000, 1, self.update_size_natoms)
        self.size_diameter = ui.SpinBox(5.0, 0, 100.0, 0.1, self.update_size_diameter)
        self.round_radio = ui.RadioButtons([_('above  '), _('below  '), _('closest  ')], ['above', 'below', 'closest'], callback=self.update)
        self.smaller_button = ui.Button(_('Smaller'), self.wulff_smaller)
        self.larger_button = ui.Button(_('Larger'), self.wulff_larger)
        rows.add(_('Choose size using:'))
        rows.add(self.size_radio)
        rows.add([_('atoms'), self.size_natoms, _(u'Å³'), self.size_diameter])
        rows.add(_('Rounding: If exact size is not possible, choose the size:'))
        rows.add(self.round_radio)
        rows.add([self.smaller_button, self.larger_button])
        self.update_gui_size()
    else:
        self.smaller_button = None
        self.larger_button = None