from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class ExpCellFilter(UnitCellFilter):
    """Modify the supercell and the atom positions."""

    def __init__(self, atoms, mask=None, cell_factor=None, hydrostatic_strain=False, constant_volume=False, scalar_pressure=0.0):
        """Create a filter that returns the atomic forces and unit cell
        stresses together, so they can simultaneously be minimized.

        The first argument, atoms, is the atoms object. The optional second
        argument, mask, is a list of booleans, indicating which of the six
        independent components of the strain are relaxed.

        - True = relax to zero
        - False = fixed, ignore this component

        Degrees of freedom are the positions in the original undeformed cell,
        plus the log of the deformation tensor (extra 3 "atoms"). This gives
        forces consistent with numerical derivatives of the potential energy
        with respect to the cell degrees of freedom.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        You can still use constraints on the atoms, e.g. FixAtoms, to control
        the relaxation of the atoms.

        >>> # this should be equivalent to the StrainFilter
        >>> atoms = Atoms(...)
        >>> atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
        >>> ecf = ExpCellFilter(atoms)

        You should not attach this ExpCellFilter object to a
        trajectory. Instead, create a trajectory for the atoms, and
        attach it to an optimizer like this:

        >>> atoms = Atoms(...)
        >>> ecf = ExpCellFilter(atoms)
        >>> qn = QuasiNewton(ecf)
        >>> traj = Trajectory('TiO2.traj', 'w', atoms)
        >>> qn.attach(traj)
        >>> qn.run(fmax=0.05)

        Helpful conversion table:

        - 0.05 eV/A^3   = 8 GPA
        - 0.003 eV/A^3  = 0.48 GPa
        - 0.0006 eV/A^3 = 0.096 GPa
        - 0.0003 eV/A^3 = 0.048 GPa
        - 0.0001 eV/A^3 = 0.02 GPa

        Additional optional arguments:

        cell_factor: (DEPRECATED)
            Retained for backwards compatibility, but no longer used.

        hydrostatic_strain: bool (default False)
            Constrain the cell by only allowing hydrostatic deformation.
            The virial tensor is replaced by np.diag([np.trace(virial)]*3).

        constant_volume: bool (default False)
            Project out the diagonal elements of the virial tensor to allow
            relaxations at constant volume, e.g. for mapping out an
            energy-volume curve.

        scalar_pressure: float (default 0.0)
            Applied pressure to use for enthalpy pV term. As above, this
            breaks energy/force consistency.

        Implementation details:

        The implementation is based on that of Christoph Ortner in JuLIP.jl:
        https://github.com/libAtoms/JuLIP.jl/blob/expcell/src/Constraints.jl#L244

        We decompose the deformation gradient as

            F = exp(U) F0
            x =  F * F0^{-1} z  = exp(U) z

        If we write the energy as a function of U we can transform the
        stress associated with a perturbation V into a derivative using a
        linear map V -> L(U, V).

        \\phi( exp(U+tV) (z+tv) ) ~ \\phi'(x) . (exp(U) v) + \\phi'(x) .
                                                    ( L(U, V) exp(-U) exp(U) z )

        where

               \\nabla E(U) : V  =  [S exp(-U)'] : L(U,V)
                                =  L'(U, S exp(-U)') : V
                                =  L(U', S exp(-U)') : V
                                =  L(U, S exp(-U)) : V     (provided U = U')

        where the : operator represents double contraction,
        i.e. A:B = trace(A'B), and

          F = deformation tensor - 3x3 matrix
          F0 = reference deformation tensor - 3x3 matrix, np.eye(3) here
          U = cell degrees of freedom used here - 3x3 matrix
          V = perturbation to cell DoFs - 3x3 matrix
          v = perturbation to position DoFs
          x = atomic positions in deformed cell
          z = atomic positions in original cell
          \\phi = potential energy
          S = stress tensor [3x3 matrix]
          L(U, V) = directional derivative of exp at U in direction V, i.e
          d/dt exp(U + t V)|_{t=0} = L(U, V)

        This means we can write

          d/dt E(U + t V)|_{t=0} = L(U, S exp (-U)) : V

        and therefore the contribution to the gradient of the energy is

          \\nabla E(U) / \\nabla U_ij =  [L(U, S exp(-U))]_ij
        """
        Filter.__init__(self, atoms, indices=range(len(atoms)))
        UnitCellFilter.__init__(self, atoms, mask, cell_factor, hydrostatic_strain, constant_volume, scalar_pressure)
        if cell_factor is not None:
            warn('cell_factor is deprecated')
        self.cell_factor = 1.0

    def get_positions(self):
        pos = UnitCellFilter.get_positions(self)
        natoms = len(self.atoms)
        pos[natoms:] = logm(self.deform_grad())
        return pos

    def set_positions(self, new, **kwargs):
        natoms = len(self.atoms)
        new2 = new.copy()
        new2[natoms:] = expm(new[natoms:])
        UnitCellFilter.set_positions(self, new2, **kwargs)

    def get_forces(self, **kwargs):
        forces = UnitCellFilter.get_forces(self, **kwargs)
        stress = self.atoms.get_stress(**kwargs)
        volume = self.atoms.get_volume()
        virial = -volume * (voigt_6_to_full_3x3_stress(stress) + np.diag([self.scalar_pressure] * 3))
        cur_deform_grad = self.deform_grad()
        cur_deform_grad_log = logm(cur_deform_grad)
        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])
        if (self.mask != 1.0).any():
            virial *= self.mask
        deform_grad_log_force_naive = virial.copy()
        Y = np.zeros((6, 6))
        Y[0:3, 0:3] = cur_deform_grad_log
        Y[3:6, 3:6] = cur_deform_grad_log
        Y[0:3, 3:6] = -virial @ expm(-cur_deform_grad_log)
        deform_grad_log_force = -expm(Y)[0:3, 3:6]
        for i1, i2 in [(0, 1), (0, 2), (1, 2)]:
            ff = 0.5 * (deform_grad_log_force[i1, i2] + deform_grad_log_force[i2, i1])
            deform_grad_log_force[i1, i2] = ff
            deform_grad_log_force[i2, i1] = ff
        all_are_equal = np.all(np.isclose(deform_grad_log_force, deform_grad_log_force_naive))
        if all_are_equal or np.sum(deform_grad_log_force * deform_grad_log_force_naive) / np.sqrt(np.sum(deform_grad_log_force ** 2) * np.sum(deform_grad_log_force_naive ** 2)) > 0.8:
            deform_grad_log_force = deform_grad_log_force_naive
        convergence_crit_stress = -(virial / volume)
        if self.constant_volume:
            dglf_trace = deform_grad_log_force.trace()
            np.fill_diagonal(deform_grad_log_force, np.diag(deform_grad_log_force) - dglf_trace / 3.0)
            ccs_trace = convergence_crit_stress.trace()
            np.fill_diagonal(convergence_crit_stress, np.diag(convergence_crit_stress) - ccs_trace / 3.0)
        natoms = len(self.atoms)
        forces[natoms:] = deform_grad_log_force
        self.stress = full_3x3_to_voigt_6_stress(convergence_crit_stress)
        return forces