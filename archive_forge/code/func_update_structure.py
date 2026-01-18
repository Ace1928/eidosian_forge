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
def update_structure(self, s):
    """Called when the user changes the structure."""
    if s != self.old_structure:
        old4 = self.fourindex
        self.fourindex = self.needs_4index[s]
        if self.fourindex != old4:
            self.default_direction_table()
        self.old_structure = s
        self.c.active = self.needs_2lat[s]
    self.update()