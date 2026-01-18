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
def update_size_diameter(self, widget=None, update=True):
    if self.size_diameter.active:
        at_vol = self.get_atomic_volume()
        n = round(np.pi / 6 * self.size_diameter.value ** 3 / at_vol)
        self.size_natoms.value = int(n)
        if update:
            self.update()