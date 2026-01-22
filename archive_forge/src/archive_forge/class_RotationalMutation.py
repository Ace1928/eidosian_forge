import numpy as np
from math import cos, sin, pi
from ase.calculators.lammpslib import convert_cell
from ase.ga.utilities import (atoms_too_close,
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation
from ase import Atoms
class RotationalMutation(OffspringCreator):
    """Mutates a candidate by applying random rotations
    to multi-atom moieties in the structure (atoms with
    the same tag are considered part of one such moiety).

    Only performs whole-molecule rotations, no internal
    rotations.

    For more information, see also:

      * `Zhu Q., Oganov A.R., Glass C.W., Stokes H.T,
        Acta Cryst. (2012), B68, 215-226.`__

        __ https://dx.doi.org/10.1107/S0108768112017466

    Parameters:

    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    n_top: int or None
        The number of atoms to optimize (None = include all).

    fraction: float
        Fraction of the moieties to be rotated.

    tags: None or list of integers
        Specifies, respectively, whether all moieties or only those
        with matching tags are eligible for rotation.

    min_angle: float
        Minimal angle (in radians) for each rotation;
        should lie in the interval [0, pi].

    test_dist_to_slab: boolean
        Whether also the distances to the slab
        should be checked to satisfy the blmin.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, blmin, n_top=None, fraction=0.33, tags=None, min_angle=1.57, test_dist_to_slab=True, rng=np.random, verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.fraction = fraction
        self.tags = tags
        self.min_angle = min_angle
        self.test_dist_to_slab = test_dist_to_slab
        self.descriptor = 'RotationalMutation'
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return (indi, 'mutation: rotational')
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), 'mutation: rotational')

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        mutant = atoms.copy()
        gather_atoms_by_tag(mutant)
        pos = mutant.get_positions()
        tags = mutant.get_tags()
        eligible_tags = tags if self.tags is None else self.tags
        indices = {}
        for tag in np.unique(tags):
            hits = np.where(tags == tag)[0]
            if len(hits) > 1 and tag in eligible_tags:
                indices[tag] = hits
        n_rot = int(np.ceil(len(indices) * self.fraction))
        chosen_tags = self.rng.choice(list(indices.keys()), size=n_rot, replace=False)
        too_close = True
        count = 0
        maxcount = 10000
        while too_close and count < maxcount:
            newpos = np.copy(pos)
            for tag in chosen_tags:
                p = np.copy(newpos[indices[tag]])
                cop = np.mean(p, axis=0)
                if len(p) == 2:
                    line = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
                    while True:
                        axis = self.rng.rand(3)
                        axis /= np.linalg.norm(axis)
                        a = np.arccos(np.dot(axis, line))
                        if np.pi / 4 < a < np.pi * 3 / 4:
                            break
                else:
                    axis = self.rng.rand(3)
                    axis /= np.linalg.norm(axis)
                angle = self.min_angle
                angle += 2 * (np.pi - self.min_angle) * self.rng.rand()
                m = get_rotation_matrix(axis, angle)
                newpos[indices[tag]] = np.dot(m, (p - cop).T).T + cop
            mutant.set_positions(newpos)
            mutant.wrap()
            too_close = atoms_too_close(mutant, self.blmin, use_tags=True)
            count += 1
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)
        if count == maxcount:
            mutant = None
        else:
            mutant = slab + mutant
        return mutant