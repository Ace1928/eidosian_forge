import numpy as np
def update_cell_and_positions(atoms, new_cell, op):
    """Helper method for transforming cell and positions of atoms object."""
    scpos = np.linalg.solve(op, atoms.get_scaled_positions().T).T
    scpos %= 1.0
    scpos %= 1.0
    atoms.set_cell(new_cell)
    atoms.set_scaled_positions(scpos)