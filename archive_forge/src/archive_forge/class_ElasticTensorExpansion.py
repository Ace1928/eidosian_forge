from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
class ElasticTensorExpansion(TensorCollection):
    """
    This class is a sequence of elastic tensors corresponding
    to an elastic tensor expansion, which can be used to
    calculate stress and energy density and inherits all
    of the list-based properties of TensorCollection
    (e. g. symmetrization, voigt conversion, etc.).
    """

    def __init__(self, c_list: Sequence) -> None:
        """
        Initialization method for ElasticTensorExpansion.

        Args:
            c_list (list or tuple): sequence of Tensor inputs or tensors from which
                the elastic tensor expansion is constructed.
        """
        c_list = [NthOrderElasticTensor(c, check_rank=4 + idx * 2) for idx, c in enumerate(c_list)]
        super().__init__(c_list)

    @classmethod
    def from_diff_fit(cls, strains, stresses, eq_stress=None, tol: float=1e-10, order=3) -> Self:
        """
        Generates an elastic tensor expansion via the fitting function
        defined below in diff_fit.
        """
        c_list = diff_fit(strains, stresses, eq_stress, order, tol)
        return cls(c_list)

    @property
    def order(self) -> int:
        """
        Order of the elastic tensor expansion, i. e. the order of the
        highest included set of elastic constants.
        """
        return self[-1].order

    def calculate_stress(self, strain) -> float:
        """
        Calculate's a given elastic tensor's contribution to the
        stress using Einstein summation.
        """
        return sum((c.calculate_stress(strain) for c in self))

    def energy_density(self, strain, convert_GPa_to_eV=True):
        """Calculates the elastic energy density due to a strain in eV/A^3 or GPa."""
        return sum((c.energy_density(strain, convert_GPa_to_eV) for c in self))

    def get_ggt(self, n, u):
        """
        Gets the Generalized Gruneisen tensor for a given
        third-order elastic tensor expansion.

        Args:
            n (3x1 array-like): normal mode direction
            u (3x1 array-like): polarization direction
        """
        gk = self[0].einsum_sequence([n, u, n, u])
        return -(2 * gk * np.outer(u, u) + self[0].einsum_sequence([n, n]) + self[1].einsum_sequence([n, u, n, u])) / (2 * gk)

    def get_tgt(self, temperature: float | None=None, structure: Structure=None, quad=None):
        """
        Gets the thermodynamic Gruneisen tensor (TGT) by via an
        integration of the GGT weighted by the directional heat
        capacity.

        See refs:
            R. N. Thurston and K. Brugger, Phys. Rev. 113, A1604 (1964).
            K. Brugger Phys. Rev. 137, A1826 (1965).

        Args:
            temperature (float): Temperature in kelvin, if not specified
                will return non-cv-normalized value
            structure (Structure): Structure to be used in directional heat
                capacity determination, only necessary if temperature
                is specified
            quadct (dict): quadrature for integration, should be
                dictionary with "points" and "weights" keys defaults
                to quadpy.sphere.Lebedev(19) as read from file
        """
        if temperature and (not structure):
            raise ValueError('If using temperature input, you must also include structure')
        quad = quad or DEFAULT_QUAD
        points = quad['points']
        weights = quad['weights']
        num, denom, c = (np.zeros((3, 3)), 0, 1)
        for p, w in zip(points, weights):
            gk = ElasticTensor(self[0]).green_kristoffel(p)
            _rho_wsquareds, us = np.linalg.eigh(gk)
            us = [u / np.linalg.norm(u) for u in np.transpose(us)]
            for u in us:
                if temperature and structure:
                    c = self.get_heat_capacity(temperature, structure, p, u)
                num += c * self.get_ggt(p, u) * w
                denom += c * w
        return SquareTensor(num / denom)

    def get_gruneisen_parameter(self, temperature=None, structure=None, quad=None):
        """
        Gets the single average gruneisen parameter from the TGT.

        Args:
            temperature (float): Temperature in kelvin, if not specified
                will return non-cv-normalized value
            structure (float): Structure to be used in directional heat
                capacity determination, only necessary if temperature
                is specified
            quadct (dict): quadrature for integration, should be
                dictionary with "points" and "weights" keys defaults
                to quadpy.sphere.Lebedev(19) as read from file
        """
        return np.trace(self.get_tgt(temperature, structure, quad)) / 3.0

    def get_heat_capacity(self, temperature, structure: Structure, n, u, cutoff=100.0):
        """
        Gets the directional heat capacity for a higher order tensor
        expansion as a function of direction and polarization.

        Args:
            temperature (float): Temperature in kelvin
            structure (float): Structure to be used in directional heat
                capacity determination
            n (3x1 array-like): direction for Cv determination
            u (3x1 array-like): polarization direction, note that
                no attempt for verification of eigenvectors is made
            cutoff (float): cutoff for scale of kt / (hbar * omega)
                if lower than this value, returns 0
        """
        k = 1.38065e-23
        kt = k * temperature
        hbar_w = 1.05457e-34 * self.omega(structure, n, u)
        if hbar_w > kt * cutoff:
            return 0.0
        c = k * (hbar_w / kt) ** 2
        c *= np.exp(hbar_w / kt) / (np.exp(hbar_w / kt) - 1) ** 2
        return c * 6.022e+23

    def omega(self, structure: Structure, n, u):
        """
        Finds directional frequency contribution to the heat
        capacity from direction and polarization.

        Args:
            structure (Structure): Structure to be used in directional heat
                capacity determination
            n (3x1 array-like): direction for Cv determination
            u (3x1 array-like): polarization direction, note that
                no attempt for verification of eigenvectors is made
        """
        l0 = np.dot(np.sum(structure.lattice.matrix, axis=0), n)
        l0 *= 1e-10
        weight = float(structure.composition.weight) * 1.66054e-27
        vol = structure.volume * 1e-30
        vel = (1000000000.0 * self[0].einsum_sequence([n, u, n, u]) / (weight / vol)) ** 0.5
        return vel / l0

    def thermal_expansion_coeff(self, structure: Structure, temperature: float, mode: Literal['dulong - petit', 'debye']='debye'):
        """
        Gets thermal expansion coefficient from third-order constants.

        Args:
            temperature (float): Temperature in kelvin, if not specified
                will return non-cv-normalized value
            structure (Structure): Structure to be used in directional heat
                capacity determination, only necessary if temperature
                is specified
            mode (str): mode for finding average heat-capacity,
                current supported modes are 'debye' and 'dulong-petit'
        """
        soec = ElasticTensor(self[0])
        v0 = structure.volume * 1e-30 / len(structure)
        if mode == 'debye':
            td = soec.debye_temperature(structure)
            t_ratio = temperature / td

            def integrand(x):
                return x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
            cv = 9 * 8.314 * t_ratio ** 3 * quad(integrand, 0, t_ratio ** (-1))[0]
        elif mode == 'dulong-petit':
            cv = 3 * 8.314
        else:
            raise ValueError(f'mode={mode!r} must be debye or dulong-petit')
        tgt = self.get_tgt(temperature, structure)
        alpha = np.einsum('ijkl,ij', soec.compliance_tensor, tgt)
        alpha *= cv / (1000000000.0 * v0 * 6.022e+23)
        return SquareTensor(alpha)

    def get_compliance_expansion(self):
        """
        Gets a compliance tensor expansion from the elastic
        tensor expansion.
        """
        if not self.order <= 4:
            raise ValueError('Compliance tensor expansion only supported for fourth-order and lower')
        ce_exp = [ElasticTensor(self[0]).compliance_tensor]
        ein_string = 'ijpq,pqrsuv,rskl,uvmn->ijklmn'
        ce_exp.append(np.einsum(ein_string, -ce_exp[-1], self[1], ce_exp[-1], ce_exp[-1]))
        if self.order == 4:
            einstring_1 = 'pqab,cdij,efkl,ghmn,abcdefgh'
            tensors_1 = [ce_exp[0]] * 4 + [self[-1]]
            temp = -np.einsum(einstring_1, *tensors_1)
            einstring_2 = 'pqab,abcdef,cdijmn,efkl'
            einstring_3 = 'pqab,abcdef,efklmn,cdij'
            einstring_4 = 'pqab,abcdef,cdijkl,efmn'
            for es in [einstring_2, einstring_3, einstring_4]:
                temp -= np.einsum(es, ce_exp[0], self[-2], ce_exp[1], ce_exp[0])
            ce_exp.append(temp)
        return TensorCollection(ce_exp)

    def get_strain_from_stress(self, stress):
        """
        Gets the strain from a stress state according
        to the compliance expansion corresponding to the
        tensor expansion.
        """
        compl_exp = self.get_compliance_expansion()
        strain = 0
        for n, compl in enumerate(compl_exp, start=1):
            strain += compl.einsum_sequence([stress] * n) / factorial(n)
        return strain

    def get_effective_ecs(self, strain, order=2):
        """
        Returns the effective elastic constants
        from the elastic tensor expansion.

        Args:
            strain (Strain or 3x3 array-like): strain condition
                under which to calculate the effective constants
            order (int): order of the ecs to be returned
        """
        ec_sum = 0
        for n, ecs in enumerate(self[order - 2:]):
            ec_sum += ecs.einsum_sequence([strain] * n) / factorial(n)
        return ec_sum

    def get_wallace_tensor(self, tau):
        """
        Gets the Wallace Tensor for determining yield strength
        criteria.

        Args:
            tau (3x3 array-like): stress at which to evaluate
                the wallace tensor
        """
        b = 0.5 * (np.einsum('ml,kn->klmn', tau, np.eye(3)) + np.einsum('km,ln->klmn', tau, np.eye(3)) + np.einsum('nl,km->klmn', tau, np.eye(3)) + np.einsum('kn,lm->klmn', tau, np.eye(3)) + -2 * np.einsum('kl,mn->klmn', tau, np.eye(3)))
        strain = self.get_strain_from_stress(tau)
        b += self.get_effective_ecs(strain)
        return b

    def get_symmetric_wallace_tensor(self, tau):
        """
        Gets the symmetrized wallace tensor for determining
        yield strength criteria.

        Args:
            tau (3x3 array-like): stress at which to evaluate
                the wallace tensor.
        """
        wallace = self.get_wallace_tensor(tau)
        return Tensor(0.5 * (wallace + np.transpose(wallace, [2, 3, 0, 1])))

    def get_stability_criteria(self, s, n):
        """
        Gets the stability criteria from the symmetric
        Wallace tensor from an input vector and stress
        value.

        Args:
            s (float): Stress value at which to evaluate
                the stability criteria
            n (3x1 array-like): direction of the applied
                stress
        """
        n = get_uvec(n)
        stress = s * np.outer(n, n)
        sym_wallace = self.get_symmetric_wallace_tensor(stress)
        return np.linalg.det(sym_wallace.voigt)

    def get_yield_stress(self, n):
        """
        Gets the yield stress for a given direction.

        Args:
            n (3x1 array-like): direction for which to find the
                yield stress
        """
        comp = root(self.get_stability_criteria, -1, args=n)
        tens = root(self.get_stability_criteria, 1, args=n)
        return (comp.x, tens.x)