import numpy as np
from math import cos, sin, pi
from ase.calculators.lammpslib import convert_cell
from ase.ga.utilities import (atoms_too_close,
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation
from ase import Atoms
class MirrorMutation(OffspringCreator):
    """A mirror mutation, as described in
    TO BE PUBLISHED.

    This mutation mirrors half of the cluster in a
    randomly oriented cutting plane discarding the other half.

    Parameters:

    blmin: Dictionary defining the minimum allowed
        distance between atoms.

    n_top: Number of atoms the GA optimizes.

    reflect: Defines if the mirrored half is also reflected
        perpendicular to the mirroring plane.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, blmin, n_top, reflect=False, rng=np.random, verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.reflect = reflect
        self.descriptor = 'MirrorMutation'
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return (indi, 'mutation: mirror')
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), 'mutation: mirror')

    def mutate(self, atoms):
        """ Do the mutation of the atoms input. """
        reflect = self.reflect
        tc = True
        slab = atoms[0:len(atoms) - self.n_top]
        top = atoms[len(atoms) - self.n_top:len(atoms)]
        num = top.numbers
        unique_types = list(set(num))
        nu = dict()
        for u in unique_types:
            nu[u] = sum(num == u)
        n_tries = 1000
        counter = 0
        changed = False
        while tc and counter < n_tries:
            counter += 1
            cand = top.copy()
            pos = cand.get_positions()
            cm = np.average(top.get_positions(), axis=0)
            theta = pi * self.rng.rand()
            phi = 2.0 * pi * self.rng.rand()
            n = (cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))
            n = np.array(n)
            D = []
            for i, p in enumerate(pos):
                d = np.dot(p - cm, n)
                D.append((i, d))
            D.sort(key=lambda x: x[1])
            nu_taken = dict()
            p_use = []
            n_use = []
            for i, d in D:
                if num[i] not in nu_taken.keys():
                    nu_taken[num[i]] = 0
                if nu_taken[num[i]] < nu[num[i]] / 2.0:
                    p_use.append(pos[i])
                    n_use.append(num[i])
                    nu_taken[num[i]] += 1
            pn = []
            for p in p_use:
                pt = p - 2.0 * np.dot(p - cm, n) * n
                if reflect:
                    pt = -pt + 2 * cm + 2 * n * np.dot(pt - cm, n)
                pn.append(pt)
            n_use.extend(n_use)
            p_use.extend(pn)
            for n in nu.keys():
                if nu[n] % 2 == 0:
                    continue
                while sum(n_use == n) > nu[n]:
                    for i in range(int(len(n_use) / 2), len(n_use)):
                        if n_use[i] == n:
                            del p_use[i]
                            del n_use[i]
                            break
                assert sum(n_use == n) == nu[n]
            for i in range(len(n_use)):
                if num[i] == n_use[i]:
                    continue
                for j in range(i + 1, len(n_use)):
                    if n_use[j] == num[i]:
                        tn = n_use[i]
                        tp = p_use[i]
                        n_use[i] = n_use[j]
                        p_use[i] = p_use[j]
                        p_use[j] = tp
                        n_use[j] = tn
            cand = Atoms(num, p_use, cell=slab.get_cell(), pbc=slab.get_pbc())
            tc = atoms_too_close(cand, self.blmin)
            if tc:
                continue
            tc = atoms_too_close_two_sets(slab, cand, self.blmin)
            if not changed and counter > n_tries // 2:
                reflect = not reflect
                changed = True
            tot = slab + cand
        if counter == n_tries:
            return None
        return tot