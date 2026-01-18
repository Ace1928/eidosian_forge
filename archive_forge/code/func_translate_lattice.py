import numpy as np
from ase.build.general_surface import surface
from ase.geometry import get_layers
from ase.symbols import string2symbols
def translate_lattice(lattice, indices, tol=10 ** (-3)):
    """translates a bulk unit cell along a normal vector given by the a set of
    miller indices to the next symetric position. This is used to control the
    termination of the surface in the smart_surface command
    Parameters:
    ==========
        lattice: Atoms object
            atoms object of the bulk unit cell
        indices: 1x3 list,tuple, or numpy array
            the miller indices you wish to cut along.
    returns:
        lattice_list: list of Atoms objects
            a list of all the different translations of the unit cell that will
            yield different terminations of a surface cut along the miller
            indices provided.
    """
    lattice_list = []
    cell = lattice.get_cell()
    pt = [0, 0, 0]
    h, k, l = indices
    millers = list(indices)
    for index, item in enumerate(millers):
        if item == 0:
            millers[index] = 10 ** 9
        elif pt == [0, 0, 0]:
            pt = list(cell[index] / float(item) / np.linalg.norm(cell[index]))
    h1, k1, l1 = millers
    N = np.array(cell[0] / h1 + cell[1] / k1 + cell[2] / l1)
    n = N / np.linalg.norm(N)
    d = [np.round(np.dot(n, a - pt) * n, 5) for a in lattice.get_scaled_positions()]
    duplicates = []
    for i, item in enumerate(d):
        g = [True for a in d[i + 1:] if np.linalg.norm(a - item) < tol]
        if g != []:
            duplicates.append(i)
    duplicates.reverse()
    for i in duplicates:
        del d[i]
    for i, item in enumerate(d):
        d[i] = np.append(item, np.dot(n, lattice.get_scaled_positions()[i] - pt))
    d = np.array(d)
    d = d[d[:, 3].argsort()]
    d = [a[:3] for a in d]
    d = list(d)
    for i in d:
        '\n        The above method gives you the boundries of between terminations that\n        will allow you to build a complete set of terminations. However, it\n        does not return all the boundries. Thus you must check both above and\n        below the boundary, and not stray too far from the boundary. If you move\n        too far away, you risk hitting another boundary you did not find.\n        '
        lattice1 = lattice.copy()
        displacement = (h * cell[0] + k * cell[1] + l * cell[2]) * (i + 10 ** (-8))
        lattice1.positions -= displacement
        lattice_list.append(lattice1)
        lattice1 = lattice.copy()
        displacement = (h * cell[0] + k * cell[1] + l * cell[2]) * (i - 10 ** (-8))
        lattice1.positions -= displacement
        lattice_list.append(lattice1)
    return lattice_list