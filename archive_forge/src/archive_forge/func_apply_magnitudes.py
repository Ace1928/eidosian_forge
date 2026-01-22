from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def apply_magnitudes(self, *args):
    atoms = self.gui.atoms.copy()
    old_mags = atoms.cell.lengths()
    new_mags = self.get_magnitudes()
    newcell = atoms.cell.copy()
    for i in range(3):
        newcell[i] *= new_mags[i] / old_mags[i]
    atoms.set_cell(newcell, scale_atoms=self.scale_atoms.var.get())
    self.gui.new_atoms(atoms)