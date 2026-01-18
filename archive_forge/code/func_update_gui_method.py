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
def update_gui_method(self, *args):
    """Switch between layer specification and Wulff construction."""
    self.update_direction_table()
    self.update_new_direction_and_size_stuff()
    if self.method.value == 'wulff':
        self.layerlabel.text = _('Surface energies (as energy/area, NOT per atom):')
    else:
        self.layerlabel.text = _('Number of layers:')
    self.update()