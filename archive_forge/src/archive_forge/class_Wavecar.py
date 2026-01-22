from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
class Wavecar:
    """
    This is a class that contains the (pseudo-) wavefunctions from VASP.

    Coefficients are read from the given WAVECAR file and the corresponding
    G-vectors are generated using the algorithm developed in WaveTrans (see
    acknowledgments below). To understand how the wavefunctions are evaluated,
    please see the evaluate_wavefunc docstring.

    It should be noted that the pseudopotential augmentation is not included in
    the WAVECAR file. As a result, some caution should be exercised when
    deriving value from this information.

    The usefulness of this class is to allow the user to do projections or band
    unfolding style manipulations of the wavefunction. An example of this can
    be seen in the work of Shen et al. 2017
    (https://doi.org/10.1103/PhysRevMaterials.1.065001).

    Attributes:
        filename (str): String of the input file (usually WAVECAR).
        vasp_type (str): String that determines VASP type the WAVECAR was generated with.
            One of 'std', 'gam', 'ncl'.
        nk (int): Number of k-points from the WAVECAR.
        nb (int): Number of bands per k-point.
        encut (float): Energy cutoff (used to define G_{cut}).
        efermi (float): Fermi energy.
        a (np.array): Primitive lattice vectors of the cell (e.g. a_1 = self.a[0, :]).
        b (np.array): Reciprocal lattice vectors of the cell (e.g. b_1 = self.b[0, :]).
        vol (float): The volume of the unit cell in real space.
        kpoints (np.array): The list of k-points read from the WAVECAR file.
        band_energy (list): The list of band eigenenergies (and corresponding occupancies) for each kpoint,
            where the first index corresponds to the index of the k-point (e.g. self.band_energy[kp]).
        Gpoints (list): The list of generated G-points for each k-point (a double list), which
            are used with the coefficients for each k-point and band to recreate
            the wavefunction (e.g. self.Gpoints[kp] is the list of G-points for
            k-point kp). The G-points depend on the k-point and reciprocal lattice
            and therefore are identical for each band at the same k-point. Each
            G-point is represented by integer multipliers (e.g. assuming
            Gpoints[kp][n] == [n_1, n_2, n_3], then
            G_n = n_1*b_1 + n_2*b_2 + n_3*b_3)
        coeffs (list): The list of coefficients for each k-point and band for reconstructing the wavefunction.
            For non-spin-polarized, the first index corresponds to the kpoint and the second corresponds to the band
            (e.g. self.coeffs[kp][b] corresponds to k-point kp and band b). For spin-polarized calculations,
            the first index is for the spin. If the calculation was non-collinear, then self.coeffs[kp][b] will have
            two columns (one for each component of the spinor).

    Acknowledgments:
        This code is based upon the Fortran program, WaveTrans, written by
        R. M. Feenstra and M. Widom from the Dept. of Physics at Carnegie
        Mellon University. To see the original work, please visit:
        https://www.andrew.cmu.edu/user/feenstra/wavetrans/

    Author: Mark Turiansky
    """

    def __init__(self, filename='WAVECAR', verbose=False, precision='normal', vasp_type=None):
        """
        Information is extracted from the given WAVECAR.

        Args:
            filename (str): input file (default: WAVECAR)
            verbose (bool): determines whether processing information is shown
            precision (str): determines how fine the fft mesh is (normal or
                accurate), only the first letter matters
            vasp_type (str): determines the VASP type that is used, allowed
                values are ['std', 'gam', 'ncl'] (only first letter is required)
        """
        self.filename = filename
        valid_types = ['std', 'gam', 'ncl']
        initials = {x[0] for x in valid_types}
        if not (vasp_type is None or vasp_type.lower()[0] in initials):
            raise ValueError(f'invalid vasp_type={vasp_type!r}, must be one of {valid_types} (we only check the first letter {initials})')
        self.vasp_type = vasp_type
        self._C = 0.262465831
        with open(self.filename, 'rb') as file:
            recl, spin, rtag = np.fromfile(file, dtype=np.float64, count=3).astype(int)
            if verbose:
                print(f'recl={recl!r}, spin={spin!r}, rtag={rtag!r}')
            recl8 = int(recl / 8)
            self.spin = spin
            valid_rtags = {45200, 45210, 53300, 53310}
            if rtag not in valid_rtags:
                raise ValueError(f'Invalid rtag={rtag!r}, must be one of {valid_rtags}')
            np.fromfile(file, dtype=np.float64, count=recl8 - 3)
            self.nk, self.nb = np.fromfile(file, dtype=np.float64, count=2).astype(int)
            self.encut = np.fromfile(file, dtype=np.float64, count=1)[0]
            self.a = np.fromfile(file, dtype=np.float64, count=9).reshape((3, 3))
            self.efermi = np.fromfile(file, dtype=np.float64, count=1)[0]
            if verbose:
                print(f'kpoints = {self.nk}, bands = {self.nb}, energy cutoff = {self.encut}, fermi energy= {self.efermi:.04f}\n')
                print(f'primitive lattice vectors = \n{self.a}')
            self.vol = np.dot(self.a[0, :], np.cross(self.a[1, :], self.a[2, :]))
            if verbose:
                print(f'volume = {self.vol}\n')
            b = np.array([np.cross(self.a[1, :], self.a[2, :]), np.cross(self.a[2, :], self.a[0, :]), np.cross(self.a[0, :], self.a[1, :])])
            b = 2 * np.pi * b / self.vol
            self.b = b
            if verbose:
                print(f'reciprocal lattice vectors = \n{b}')
                print(f'reciprocal lattice vector magnitudes = \n{np.linalg.norm(b, axis=1)}\n')
            self._generate_nbmax()
            if verbose:
                print(f'max number of G values = {self._nbmax}\n\n')
            self.ng = self._nbmax * 3 if precision.lower()[0] == 'n' else self._nbmax * 4
            np.fromfile(file, dtype=np.float64, count=recl8 - 13)
            self.Gpoints = [None for _ in range(self.nk)]
            self.kpoints = []
            if spin == 2:
                self.coeffs = [[[None for i in range(self.nb)] for j in range(self.nk)] for _ in range(spin)]
                self.band_energy = [[] for _ in range(spin)]
            else:
                self.coeffs = [[None for i in range(self.nb)] for j in range(self.nk)]
                self.band_energy = []
            for ispin in range(spin):
                if verbose:
                    print(f'reading spin {ispin}')
                for ink in range(self.nk):
                    nplane = int(np.fromfile(file, dtype=np.float64, count=1)[0])
                    kpoint = np.fromfile(file, dtype=np.float64, count=3)
                    if ispin == 0:
                        self.kpoints.append(kpoint)
                    else:
                        assert_allclose(self.kpoints[ink], kpoint)
                    if verbose:
                        print(f'kpoint {ink: 4} with {nplane: 5} plane waves at {kpoint}')
                    enocc = np.fromfile(file, dtype=np.float64, count=3 * self.nb).reshape((self.nb, 3))
                    if spin == 2:
                        self.band_energy[ispin].append(enocc)
                    else:
                        self.band_energy.append(enocc)
                    if verbose:
                        print('enocc =\n', enocc[:, [0, 2]])
                    np.fromfile(file, dtype=np.float64, count=(recl8 - 4 - 3 * self.nb) % recl8)
                    if self.vasp_type is None:
                        self.Gpoints[ink], extra_gpoints, extra_coeff_inds = self._generate_G_points(kpoint, gamma=True)
                        if len(self.Gpoints[ink]) == nplane:
                            self.vasp_type = 'gam'
                        else:
                            self.Gpoints[ink], extra_gpoints, extra_coeff_inds = self._generate_G_points(kpoint, gamma=False)
                            self.vasp_type = 'std' if len(self.Gpoints[ink]) == nplane else 'ncl'
                        if verbose:
                            print(f'\ndetermined self.vasp_type = {self.vasp_type!r}\n')
                    else:
                        self.Gpoints[ink], extra_gpoints, extra_coeff_inds = self._generate_G_points(kpoint, gamma=self.vasp_type.lower()[0] == 'g')
                    if len(self.Gpoints[ink]) != nplane and 2 * len(self.Gpoints[ink]) != nplane:
                        raise ValueError(f'Incorrect vasp_type={vasp_type!r}. Please open an issue if you are certain this WAVECAR was generated with the given vasp_type.')
                    self.Gpoints[ink] = np.array(self.Gpoints[ink] + extra_gpoints, dtype=np.float64)
                    for inb in range(self.nb):
                        if rtag in (45200, 53300):
                            data = np.fromfile(file, dtype=np.complex64, count=nplane)
                            np.fromfile(file, dtype=np.float64, count=recl8 - nplane)
                        elif rtag in (45210, 53310):
                            data = np.fromfile(file, dtype=np.complex128, count=nplane)
                            np.fromfile(file, dtype=np.float64, count=recl8 - 2 * nplane)
                        extra_coeffs = []
                        if len(extra_coeff_inds) > 0:
                            for G_ind in extra_coeff_inds:
                                data[G_ind] /= np.sqrt(2)
                                extra_coeffs.append(np.conj(data[G_ind]))
                        if spin == 2:
                            self.coeffs[ispin][ink][inb] = np.array(list(data) + extra_coeffs, dtype=np.complex64)
                        else:
                            self.coeffs[ink][inb] = np.array(list(data) + extra_coeffs, dtype=np.complex128)
                        if self.vasp_type.lower()[0] == 'n':
                            self.coeffs[ink][inb].shape = (2, nplane // 2)

    def _generate_nbmax(self) -> None:
        """
        Helper function that determines maximum number of b vectors for
        each direction.

        This algorithm is adapted from WaveTrans (see Class docstring). There
        should be no reason for this function to be called outside of
        initialization.
        """
        bmag = np.linalg.norm(self.b, axis=1)
        b = self.b
        phi12 = np.arccos(np.dot(b[0, :], b[1, :]) / (bmag[0] * bmag[1]))
        sphi123 = np.dot(b[2, :], np.cross(b[0, :], b[1, :])) / (bmag[2] * np.linalg.norm(np.cross(b[0, :], b[1, :])))
        nbmaxA = np.sqrt(self.encut * self._C) / bmag
        nbmaxA[0] /= np.abs(np.sin(phi12))
        nbmaxA[1] /= np.abs(np.sin(phi12))
        nbmaxA[2] /= np.abs(sphi123)
        nbmaxA += 1
        phi13 = np.arccos(np.dot(b[0, :], b[2, :]) / (bmag[0] * bmag[2]))
        sphi123 = np.dot(b[1, :], np.cross(b[0, :], b[2, :])) / (bmag[1] * np.linalg.norm(np.cross(b[0, :], b[2, :])))
        nbmaxB = np.sqrt(self.encut * self._C) / bmag
        nbmaxB[0] /= np.abs(np.sin(phi13))
        nbmaxB[1] /= np.abs(sphi123)
        nbmaxB[2] /= np.abs(np.sin(phi13))
        nbmaxB += 1
        phi23 = np.arccos(np.dot(b[1, :], b[2, :]) / (bmag[1] * bmag[2]))
        sphi123 = np.dot(b[0, :], np.cross(b[1, :], b[2, :])) / (bmag[0] * np.linalg.norm(np.cross(b[1, :], b[2, :])))
        nbmaxC = np.sqrt(self.encut * self._C) / bmag
        nbmaxC[0] /= np.abs(sphi123)
        nbmaxC[1] /= np.abs(np.sin(phi23))
        nbmaxC[2] /= np.abs(np.sin(phi23))
        nbmaxC += 1
        self._nbmax = np.max([nbmaxA, nbmaxB, nbmaxC], axis=0).astype(int)

    def _generate_G_points(self, kpoint: np.ndarray, gamma: bool=False) -> tuple[list, list, list]:
        """
        Helper function to generate G-points based on nbmax.

        This function iterates over possible G-point values and determines
        if the energy is less than G_{cut}. Valid values are appended to
        the output array. This function should not be called outside of
        initialization.

        Args:
            kpoint (np.array): the array containing the current k-point value
            gamma (bool): determines if G points for gamma-point only executable
                          should be generated

        Returns:
            a list containing valid G-points
        """
        kmax = self._nbmax[0] + 1 if gamma else 2 * self._nbmax[0] + 1
        gpoints = []
        extra_gpoints = []
        extra_coeff_inds = []
        G_ind = 0
        for i in range(2 * self._nbmax[2] + 1):
            i3 = i - 2 * self._nbmax[2] - 1 if i > self._nbmax[2] else i
            for j in range(2 * self._nbmax[1] + 1):
                j2 = j - 2 * self._nbmax[1] - 1 if j > self._nbmax[1] else j
                for k in range(kmax):
                    k1 = k - 2 * self._nbmax[0] - 1 if k > self._nbmax[0] else k
                    if gamma and (k1 == 0 and j2 < 0 or (k1 == 0 and j2 == 0 and (i3 < 0))):
                        continue
                    G = np.array([k1, j2, i3])
                    v = kpoint + G
                    g = np.linalg.norm(np.dot(v, self.b))
                    E = g ** 2 / self._C
                    if self.encut > E:
                        gpoints.append(G)
                        if gamma and (k1, j2, i3) != (0, 0, 0):
                            extra_gpoints.append(-G)
                            extra_coeff_inds.append(G_ind)
                        G_ind += 1
        return (gpoints, extra_gpoints, extra_coeff_inds)

    def evaluate_wavefunc(self, kpoint: int, band: int, r: np.ndarray, spin: int=0, spinor: int=0) -> np.complex64:
        """
        Evaluates the wavefunction for a given position, r.

        The wavefunction is given by the k-point and band. It is evaluated
        at the given position by summing over the components. Formally,

        \\psi_n^k (r) = \\sum_{i=1}^N c_i^{n,k} \\exp (i (k + G_i^{n,k}) \\cdot r)

        where \\psi_n^k is the wavefunction for the nth band at k-point k, N is
        the number of plane waves, c_i^{n,k} is the ith coefficient that
        corresponds to the nth band and k-point k, and G_i^{n,k} is the ith
        G-point corresponding to k-point k.

        NOTE: This function is very slow; a discrete fourier transform is the
        preferred method of evaluation (see Wavecar.fft_mesh).

        Args:
            kpoint (int): the index of the kpoint where the wavefunction will be evaluated
            band (int): the index of the band where the wavefunction will be evaluated
            r (np.array): the position where the wavefunction will be evaluated
            spin (int): spin index for the desired wavefunction (only for
                ISPIN = 2, default = 0)
            spinor (int): component of the spinor that is evaluated (only used
                if vasp_type == 'ncl')

        Returns:
            a complex value corresponding to the evaluation of the wavefunction
        """
        v = self.Gpoints[kpoint] + self.kpoints[kpoint]
        u = np.dot(np.dot(v, self.b), r)
        if self.vasp_type.lower()[0] == 'n':
            c = self.coeffs[kpoint][band][spinor, :]
        elif self.spin == 2:
            c = self.coeffs[spin][kpoint][band]
        else:
            c = self.coeffs[kpoint][band]
        return np.sum(np.dot(c, np.exp(1j * u, dtype=np.complex64))) / np.sqrt(self.vol)

    def fft_mesh(self, kpoint: int, band: int, spin: int=0, spinor: int=0, shift: bool=True) -> np.ndarray:
        """
        Places the coefficients of a wavefunction onto an fft mesh.

        Once the mesh has been obtained, a discrete fourier transform can be
        used to obtain real-space evaluation of the wavefunction. The output
        of this function can be passed directly to numpy's fft function. For
        example:

            mesh = Wavecar('WAVECAR').fft_mesh(kpoint, band)
            evals = np.fft.ifftn(mesh)

        Args:
            kpoint (int): the index of the kpoint where the wavefunction will be evaluated
            band (int): the index of the band where the wavefunction will be evaluated
            spin (int): the spin of the wavefunction for the desired
                wavefunction (only for ISPIN = 2, default = 0)
            spinor (int): component of the spinor that is evaluated (only used
                if vasp_type == 'ncl')
            shift (bool): determines if the zero frequency coefficient is
                placed at index (0, 0, 0) or centered

        Returns:
            a numpy ndarray representing the 3D mesh of coefficients
        """
        if self.vasp_type.lower()[0] == 'n':
            tcoeffs = self.coeffs[kpoint][band][spinor, :]
        elif self.spin == 2:
            tcoeffs = self.coeffs[spin][kpoint][band]
        else:
            tcoeffs = self.coeffs[kpoint][band]
        mesh = np.zeros(tuple(self.ng), dtype=np.complex128)
        for gp, coeff in zip(self.Gpoints[kpoint], tcoeffs):
            t = tuple(gp.astype(int) + (self.ng / 2).astype(int))
            mesh[t] = coeff
        if shift:
            return np.fft.ifftshift(mesh)
        return mesh

    def get_parchg(self, poscar: Poscar, kpoint: int, band: int, spin: int | None=None, spinor: int | None=None, phase: bool=False, scale: int=2) -> Chgcar:
        """
        Generates a Chgcar object, which is the charge density of the specified
        wavefunction.

        This function generates a Chgcar object with the charge density of the
        wavefunction specified by band and kpoint (and spin, if the WAVECAR
        corresponds to a spin-polarized calculation). The phase tag is a
        feature that is not present in VASP. For a real wavefunction, the phase
        tag being turned on means that the charge density is multiplied by the
        sign of the wavefunction at that point in space. A warning is generated
        if the phase tag is on and the chosen kpoint is not Gamma.

        Note: Augmentation from the PAWs is NOT included in this function. The
        maximal charge density will differ from the PARCHG from VASP, but the
        qualitative shape of the charge density will match.

        Args:
            poscar (pymatgen.io.vasp.inputs.Poscar): Poscar object that has the
                structure associated with the WAVECAR file
            kpoint (int): the index of the kpoint for the wavefunction
            band (int): the index of the band for the wavefunction
            spin (int): optional argument to specify the spin. If the Wavecar
                has ISPIN = 2, spin is None generates a Chgcar with total spin
                and magnetization, and spin == {0, 1} specifies just the spin
                up or down component.
            spinor (int): optional argument to specify the spinor component
                for noncollinear data wavefunctions (allowed values of None,
                0, or 1)
            phase (bool): flag to determine if the charge density is multiplied
                by the sign of the wavefunction. Only valid for real
                wavefunctions.
            scale (int): scaling for the FFT grid. The default value of 2 is at
                least as fine as the VASP default.

        Returns:
            a pymatgen.io.vasp.outputs.Chgcar object
        """
        if phase and (not np.all(self.kpoints[kpoint] == 0.0)):
            warnings.warn("phase is True should only be used for the Gamma kpoint! I hope you know what you're doing!")
        temp_ng = self.ng
        self.ng = self.ng * scale
        N = np.prod(self.ng)
        data = {}
        if self.spin == 2:
            if spin is not None:
                wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spin=spin)) * N
                den = np.abs(np.conj(wfr) * wfr)
                if phase:
                    den = np.sign(np.real(wfr)) * den
                data['total'] = den
            else:
                wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spin=0)) * N
                denup = np.abs(np.conj(wfr) * wfr)
                wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spin=1)) * N
                dendn = np.abs(np.conj(wfr) * wfr)
                data['total'] = denup + dendn
                data['diff'] = denup - dendn
        else:
            if spinor is not None:
                wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spinor=spinor)) * N
                den = np.abs(np.conj(wfr) * wfr)
            else:
                wfr = np.fft.ifftn(self.fft_mesh(kpoint, band, spinor=0)) * N
                wfr_t = np.fft.ifftn(self.fft_mesh(kpoint, band, spinor=1)) * N
                den = np.abs(np.conj(wfr) * wfr)
                den += np.abs(np.conj(wfr_t) * wfr_t)
            if phase and (not (self.vasp_type.lower()[0] == 'n' and spinor is None)):
                den = np.sign(np.real(wfr)) * den
            data['total'] = den
        self.ng = temp_ng
        return Chgcar(poscar, data)

    def write_unks(self, directory: str) -> None:
        """
        Write the UNK files to the given directory.

        Writes the cell-periodic part of the bloch wavefunctions from the
        WAVECAR file to each of the UNK files. There will be one UNK file for
        each of the kpoints in the WAVECAR file.

        Note:
            wannier90 expects the full kpoint grid instead of the symmetry-
            reduced one that VASP stores the wavefunctions on. You should run
            a nscf calculation with ISYM=0 to obtain the correct grid.

        Args:
            directory (str): directory where the UNK files are written
        """
        out_dir = Path(directory).expanduser()
        if not out_dir.exists():
            out_dir.mkdir(parents=False)
        elif not out_dir.is_dir():
            raise ValueError('invalid directory')
        N = np.prod(self.ng)
        for ik in range(self.nk):
            fname = f'UNK{ik + 1:05d}.'
            if self.vasp_type.lower()[0] == 'n':
                data = np.empty((self.nb, 2, *self.ng), dtype=np.complex128)
                for ib in range(self.nb):
                    data[ib, 0, :, :, :] = np.fft.ifftn(self.fft_mesh(ik, ib, spinor=0)) * N
                    data[ib, 1, :, :, :] = np.fft.ifftn(self.fft_mesh(ik, ib, spinor=1)) * N
                Unk(ik + 1, data).write_file(str(out_dir / (fname + 'NC')))
            else:
                data = np.empty((self.nb, *self.ng), dtype=np.complex128)
                for ispin in range(self.spin):
                    for ib in range(self.nb):
                        data[ib, :, :, :] = np.fft.ifftn(self.fft_mesh(ik, ib, spin=ispin)) * N
                    Unk(ik + 1, data).write_file(str(out_dir / f'{fname}{ispin + 1}'))