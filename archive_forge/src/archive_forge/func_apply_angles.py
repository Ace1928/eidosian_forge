from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def apply_angles(self, *args):
    atoms = self.gui.atoms.copy()
    cell_data = atoms.cell.cellpar()
    cell_data[3:7] = [self.angles[0].value, self.angles[1].value, self.angles[2].value]
    atoms.set_cell(cell_data, scale_atoms=self.scale_atoms.var.get())
    self.gui.new_atoms(atoms)