import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
class SoftMutation(OffspringCreator):
    """Mutates the structure by displacing it along the lowest
    (nonzero) frequency modes found by vibrational analysis, as in:

    `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632`__

    __ https://dx.doi.org/10.1016/j.cpc.2010.06.007

    As in the reference above, the next-lowest mode is used if the
    structure has already been softmutated along the current-lowest
    mode. This mutation hence acts in a deterministic way, in contrast
    to most other genetic operators.

    If you find this implementation useful in your work,
    please consider citing:

    `Van den Bossche, Gronbeck, Hammer, J. Chem. Theory Comput. 14 (2018)`__

    __ https://dx.doi.org/10.1021/acs.jctc.8b00039

    in addition to the paper mentioned above.

    Parameters:

    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    bounds: list
        Lower and upper limits (in Angstrom) for the largest
        atomic displacement in the structure. For a given mode,
        the algorithm starts at zero amplitude and increases
        it until either blmin is violated or the largest
        displacement exceeds the provided upper bound).
        If the largest displacement in the resulting structure
        is lower than the provided lower bound, the mutant is
        considered too similar to the parent and None is
        returned.

    calculator: ASE calculator object
        The calculator to be used in the vibrational
        analysis. The default is to use a calculator
        based on pairwise harmonic potentials with force
        constants from the "bond electronegativity"
        model described in the reference above.
        Any calculator with a working :func:`get_forces()`
        method will work.

    rcut: float
        Cutoff radius in Angstrom for the pairwise harmonic
        potentials.

    used_modes_file: str or None
        Name of json dump file where previously used
        modes will be stored (and read). If None,
        no such file will be used. Default is to use
        the filename 'used_modes.json'.

    use_tags: boolean
        Whether to use the atomic tags to preserve molecular identity.
    """

    def __init__(self, blmin, bounds=[0.5, 2.0], calculator=BondElectroNegativityModel, rcut=10.0, used_modes_file='used_modes.json', use_tags=False, verbose=False):
        OffspringCreator.__init__(self, verbose)
        self.blmin = blmin
        self.bounds = bounds
        self.calc = calculator
        self.rcut = rcut
        self.used_modes_file = used_modes_file
        self.use_tags = use_tags
        self.descriptor = 'SoftMutation'
        self.used_modes = {}
        if self.used_modes_file is not None:
            try:
                self.read_used_modes(self.used_modes_file)
            except IOError:
                pass

    def _get_hessian(self, atoms, dx):
        """Returns the Hessian matrix d2E/dxi/dxj using a first-order
        central difference scheme with displacements dx.
        """
        N = len(atoms)
        pos = atoms.get_positions()
        hessian = np.zeros((3 * N, 3 * N))
        for i in range(3 * N):
            row = np.zeros(3 * N)
            for direction in [-1, 1]:
                disp = np.zeros(3)
                disp[i % 3] = direction * dx
                pos_disp = np.copy(pos)
                pos_disp[i // 3] += disp
                atoms.set_positions(pos_disp)
                f = atoms.get_forces()
                row += -1 * direction * f.flatten()
            row /= 2.0 * dx
            hessian[i] = row
        hessian += np.copy(hessian).T
        hessian *= 0.5
        atoms.set_positions(pos)
        return hessian

    def _calculate_normal_modes(self, atoms, dx=0.02, massweighing=False):
        """Performs the vibrational analysis."""
        hessian = self._get_hessian(atoms, dx)
        if massweighing:
            m = np.array([np.repeat(atoms.get_masses() ** (-0.5), 3)])
            hessian *= m * m.T
        eigvals, eigvecs = np.linalg.eigh(hessian)
        modes = {eigval: eigvecs[:, i] for i, eigval in enumerate(eigvals)}
        return modes

    def animate_mode(self, atoms, mode, nim=30, amplitude=1.0):
        """Returns an Atoms object showing an animation of the mode."""
        pos = atoms.get_positions()
        mode = mode.reshape(np.shape(pos))
        animation = []
        for i in range(nim):
            newpos = pos + amplitude * mode * np.sin(i * 2 * np.pi / nim)
            image = atoms.copy()
            image.positions = newpos
            animation.append(image)
        return animation

    def read_used_modes(self, filename):
        """Read used modes from json file."""
        with open(filename, 'r') as fd:
            modes = json.load(fd)
            self.used_modes = {int(k): modes[k] for k in modes}
        return

    def write_used_modes(self, filename):
        """Dump used modes to json file."""
        with open(filename, 'w') as fd:
            json.dump(self.used_modes, fd)
        return

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return (indi, 'mutation: soft')
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), 'mutation: soft')

    def mutate(self, atoms):
        """Does the actual mutation."""
        a = atoms.copy()
        if inspect.isclass(self.calc):
            assert issubclass(self.calc, PairwiseHarmonicPotential)
            calc = self.calc(atoms, rcut=self.rcut)
        else:
            calc = self.calc
        a.calc = calc
        if self.use_tags:
            a = TagFilter(a)
        pos = a.get_positions()
        modes = self._calculate_normal_modes(a)
        keys = np.array(sorted(modes))
        index = 3
        confid = atoms.info['confid']
        if confid in self.used_modes:
            while index in self.used_modes[confid]:
                index += 1
            self.used_modes[confid].append(index)
        else:
            self.used_modes[confid] = [index]
        if self.used_modes_file is not None:
            self.write_used_modes(self.used_modes_file)
        key = keys[index]
        mode = modes[key].reshape(np.shape(pos))
        mutant = atoms.copy()
        amplitude = 0.0
        increment = 0.1
        direction = 1
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))

        def expand(atoms, positions):
            if isinstance(atoms, TagFilter):
                a.set_positions(positions)
                return a.atoms.get_positions()
            else:
                return positions
        while amplitude * largest_norm < self.bounds[1]:
            pos_new = pos + direction * amplitude * mode
            pos_new = expand(a, pos_new)
            mutant.set_positions(pos_new)
            mutant.wrap()
            too_close = atoms_too_close(mutant, self.blmin, use_tags=self.use_tags)
            if too_close:
                amplitude -= increment
                pos_new = pos + direction * amplitude * mode
                pos_new = expand(a, pos_new)
                mutant.set_positions(pos_new)
                mutant.wrap()
                break
            if direction == 1:
                direction = -1
            else:
                direction = 1
                amplitude += increment
        if amplitude * largest_norm < self.bounds[0]:
            mutant = None
        return mutant