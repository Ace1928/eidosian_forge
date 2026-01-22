import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
class PrimitiveNeighborList:
    """Neighbor list that works without Atoms objects.

    This is less fancy, but can be used to avoid conversions between
    scaled and non-scaled coordinates which may affect cell offsets
    through rounding errors.
    """

    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True, bothways=False, use_scaled_positions=False):
        self.cutoffs = np.asarray(cutoffs) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update(self, pbc, cell, coordinates):
        """Make sure the list is up to date."""
        if self.nupdates == 0:
            self.build(pbc, cell, coordinates)
            return True
        if (self.pbc != pbc).any() or (self.cell != cell).any() or ((self.coordinates - coordinates) ** 2).sum(1).max() > self.skin ** 2:
            self.build(pbc, cell, coordinates)
            return True
        return False

    def build(self, pbc, cell, coordinates):
        """Build the list.

        Coordinates are taken to be scaled or not according
        to self.use_scaled_positions.
        """
        self.pbc = pbc = np.array(pbc, copy=True)
        self.cell = cell = Cell(cell)
        self.coordinates = coordinates = np.array(coordinates, copy=True)
        if len(self.cutoffs) != len(coordinates):
            raise ValueError('Wrong number of cutoff radii: {0} != {1}'.format(len(self.cutoffs), len(coordinates)))
        if len(self.cutoffs) > 0:
            rcmax = self.cutoffs.max()
        else:
            rcmax = 0.0
        if self.use_scaled_positions:
            positions0 = cell.cartesian_positions(coordinates)
        else:
            positions0 = coordinates
        rcell, op = minkowski_reduce(cell, pbc)
        positions = wrap_positions(positions0, rcell, pbc=pbc, eps=0)
        natoms = len(positions)
        self.nneighbors = 0
        self.npbcneighbors = 0
        self.neighbors = [np.empty(0, int) for a in range(natoms)]
        self.displacements = [np.empty((0, 3), int) for a in range(natoms)]
        self.nupdates += 1
        if natoms == 0:
            return
        N = []
        ircell = np.linalg.pinv(rcell)
        for i in range(3):
            if self.pbc[i]:
                v = ircell[:, i]
                h = 1 / np.linalg.norm(v)
                n = int(2 * rcmax / h) + 1
            else:
                n = 0
            N.append(n)
        tree = cKDTree(positions, copy_data=True)
        offsets = cell.scaled_positions(positions - positions0)
        offsets = offsets.round().astype(int)
        for n1, n2, n3 in itertools.product(range(0, N[0] + 1), range(-N[1], N[1] + 1), range(-N[2], N[2] + 1)):
            if n1 == 0 and (n2 < 0 or (n2 == 0 and n3 < 0)):
                continue
            displacement = (n1, n2, n3) @ rcell
            for a in range(natoms):
                indices = tree.query_ball_point(positions[a] - displacement, r=self.cutoffs[a] + rcmax)
                if not len(indices):
                    continue
                indices = np.array(indices)
                delta = positions[indices] + displacement - positions[a]
                cutoffs = self.cutoffs[indices] + self.cutoffs[a]
                i = indices[np.linalg.norm(delta, axis=1) < cutoffs]
                if n1 == 0 and n2 == 0 and (n3 == 0):
                    if self.self_interaction:
                        i = i[i >= a]
                    else:
                        i = i[i > a]
                self.nneighbors += len(i)
                self.neighbors[a] = np.concatenate((self.neighbors[a], i))
                disp = (n1, n2, n3) @ op + offsets[i] - offsets[a]
                self.npbcneighbors += disp.any(1).sum()
                self.displacements[a] = np.concatenate((self.displacements[a], disp))
        if self.bothways:
            neighbors2 = [[] for a in range(natoms)]
            displacements2 = [[] for a in range(natoms)]
            for a in range(natoms):
                for b, disp in zip(self.neighbors[a], self.displacements[a]):
                    neighbors2[b].append(a)
                    displacements2[b].append(-disp)
            for a in range(natoms):
                nbs = np.concatenate((self.neighbors[a], neighbors2[a]))
                disp = np.array(list(self.displacements[a]) + displacements2[a])
                self.neighbors[a] = nbs.astype(int)
                self.displacements[a] = disp.astype(int).reshape((-1, 3))
        if self.sorted:
            for a, i in enumerate(self.neighbors):
                mask = i < a
                if mask.any():
                    j = i[mask]
                    offsets = self.displacements[a][mask]
                    for b, offset in zip(j, offsets):
                        self.neighbors[b] = np.concatenate((self.neighbors[b], [a]))
                        self.displacements[b] = np.concatenate((self.displacements[b], [-offset]))
                    mask = np.logical_not(mask)
                    self.neighbors[a] = self.neighbors[a][mask]
                    self.displacements[a] = self.displacements[a][mask]

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this::

          indices, offsets = nl.get_neighbors(42)
          for i, offset in zip(indices, offsets):
              print(atoms.positions[i] + offset @ atoms.get_cell())

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""
        return (self.neighbors[a], self.displacements[a])