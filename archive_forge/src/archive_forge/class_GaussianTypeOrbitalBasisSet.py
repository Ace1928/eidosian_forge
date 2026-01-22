from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@dataclass
class GaussianTypeOrbitalBasisSet(AtomicMetadata):
    """
    Model definition of a GTO basis set.

    Attributes:
        info: Cardinality of this basis
        nset: Number of exponent sets
        n: Principle quantum number for each set
        lmax: Maximum angular momentum quantum number for each set
        lmin: Minimum angular momentum quantum number for each set
        nshell: Number of shells for angular momentum l for each set
        exponents: Exponents for each set
        coefficients: Contraction coefficients for each set. Dict[exp->l->shell]
    """
    info: BasisInfo | None = None
    nset: int | None = None
    n: list[int] | None = None
    lmax: list[int] | None = None
    lmin: list[int] | None = None
    nshell: list[dict[int, int]] | None = None
    exponents: list[list[float]] | None = None
    coefficients: list[dict[int, dict[int, dict[int, float]]]] | None = None

    def __post_init__(self) -> None:
        if self.info and self.potential == 'All Electron' and self.element:
            self.info.electrons = self.element.Z
        if self.name == 'ALLELECTRON':
            self.name = 'ALL'

        def cast(d):
            new = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    v = cast(v)
                new[int(k)] = v
            return new
        if self.nshell:
            self.nshell = [cast(n) for n in self.nshell]
        if self.coefficients:
            self.coefficients = [cast(c) for c in self.coefficients]

    def get_keyword(self) -> Keyword:
        """Convert basis to keyword object."""
        if not self.name:
            raise ValueError('No name attribute. Cannot create keyword')
        vals: Any = []
        if self.info and self.info.admm:
            vals.append('AUX_FIT')
        vals.append(self.name)
        return Keyword('BASIS_SET', *vals)

    @property
    def nexp(self):
        """Number of exponents."""
        return [len(exp) for exp in self.exponents]

    @typing.no_type_check
    def get_str(self) -> str:
        """Get standard cp2k GTO formatted string."""
        if self.info is None or self.nset is None or self.n is None or (self.lmax is None) or (self.lmin is None) or (self.nshell is None) or (self.exponents is None) or (self.coefficients is None):
            raise ValueError('Must have all attributes defined to get string representation')
        out = f'{self.element} {self.name} {' '.join(self.alias_names)}\n'
        out += f'{self.nset}\n'
        for set_index in range(self.nset):
            out += f'{self.n[set_index]} {self.lmin[set_index]} {self.lmax[set_index]} {self.nexp[set_index]} {' '.join(map(str, self.nshell[set_index].values()))}\n'
            for exp in self.coefficients[set_index]:
                out += f'\t {self.exponents[set_index][exp]: .14f} '
                for ll in self.coefficients[set_index][exp]:
                    for shell in self.coefficients[set_index][exp][ll]:
                        out += f'{self.coefficients[set_index][exp][ll][shell]: .14f} '
                out += '\n'
        return out

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Read from standard cp2k GTO formatted string."""
        lines = [line for line in string.split('\n') if line]
        firstline = lines[0].split()
        element = Element(firstline[0])
        names = firstline[1:]
        name, aliases = (names[0], names[1:])
        _info = BasisInfo.from_str(name).as_dict()
        for alias in aliases:
            for k, v in BasisInfo.from_str(alias).as_dict().items():
                if _info[k] is None:
                    _info[k] = v
        info = BasisInfo.from_dict(_info)
        potential: Literal['All Electron', 'Pseudopotential']
        if any(('ALL' in x for x in [name, *aliases])):
            info.electrons = element.Z
            potential = 'All Electron'
        else:
            potential = 'Pseudopotential'
        nset = int(lines[1].split()[0])
        n = []
        lmin = []
        lmax = []
        nshell = []
        exponents: list[list[float]] = []
        coefficients: list[dict[int, dict[int, dict[int, float]]]] = []
        line_index = 2
        for set_index in range(nset):
            setinfo = lines[line_index].split()
            _n, _lmin, _lmax, _nexp = map(int, setinfo[:4])
            n.append(_n)
            lmin.append(_lmin)
            lmax.append(_lmax)
            _nshell = map(int, setinfo[4:])
            nshell.append({ll: int(next(_nshell, 0)) for ll in range(_lmin, _lmax + 1)})
            exponents.append([])
            coefficients.append({i: {ll: {} for ll in range(_lmin, _lmax + 1)} for i in range(_nexp)})
            line_index += 1
            for ii in range(_nexp):
                line = lines[line_index].split()
                exponents[set_index].append(float(line[0]))
                coeffs = list(map(float, line[1:]))
                jj = 0
                for ll in range(_lmin, _lmax + 1):
                    for shell in range(nshell[set_index][ll]):
                        coefficients[set_index][ii][ll][shell] = coeffs[jj]
                        jj += 1
                line_index += 1
        return cls(element=element, name=name, alias_names=aliases, info=info, potential=potential, nset=nset, n=n, lmin=lmin, lmax=lmax, nshell=nshell, exponents=exponents, coefficients=coefficients)