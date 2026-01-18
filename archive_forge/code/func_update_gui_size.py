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
def update_gui_size(self, widget=None):
    """Update gui when the cluster size specification changes."""
    self.size_natoms.active = self.size_radio.value == 'natoms'
    self.size_diameter.active = self.size_radio.value == 'diameter'