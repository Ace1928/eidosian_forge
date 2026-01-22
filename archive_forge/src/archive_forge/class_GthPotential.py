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
class GthPotential(AtomicMetadata):
    """
    Representation of GTH-type (pseudo)potential.

    Attributes:
        info: Info about this potential
        n_elecs: Number of electrons for each quantum number
        r_loc: Radius of local projectors
        nexp_ppl: Number of the local pseudopotential functions
        c_exp_ppl: Sequence = field(None, description="Coefficients of the local pseudopotential functions
        radii: Radius of the nonlocal part for angular momentum quantum number l defined by the Gaussian
            function exponents alpha_prj_ppnl
        nprj: Number of projectors
        nprj_ppnl: Number of the non-local projectors for the angular momentum quantum number
        hprj_ppnl: Coefficients of the non-local projector functions. Coeff ij for ang momentum l
        )
    """
    info: PotentialInfo
    n_elecs: dict[int, int] | None = None
    r_loc: float | None = None
    nexp_ppl: int | None = None
    c_exp_ppl: Sequence | None = None
    radii: dict[int, float] | None = None
    nprj: int | None = None
    nprj_ppnl: dict[int, int] | None = None
    hprj_ppnl: dict[int, dict[int, dict[int, float]]] | None = None

    def __post_init__(self) -> None:
        if self.potential == 'All Electron' and self.element:
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
        if self.n_elecs:
            self.n_elecs = cast(self.n_elecs)
        if self.radii:
            self.radii = cast(self.radii)
        if self.nprj_ppnl:
            self.nprj_ppnl = cast(self.nprj_ppnl)
        if self.hprj_ppnl:
            self.hprj_ppnl = cast(self.hprj_ppnl)

    def get_keyword(self) -> Keyword:
        """Get keyword object for the potential."""
        if self.name is None:
            raise ValueError('Cannot get keyword without name attribute')
        return Keyword('POTENTIAL', self.name)

    def get_section(self) -> Section:
        """Convert model to a GTH-formatted section object for input files."""
        if self.name is None:
            raise ValueError('Cannot get section without name attribute')
        keywords = {'POTENTIAL': Keyword('', self.get_str())}
        return Section(name=self.name, section_parameters=None, subsections=None, description='Manual definition of GTH Potential', keywords=keywords)

    @classmethod
    def from_section(cls, section: Section) -> Self:
        """Extract GTH-formatted string from a section and convert it to model."""
        sec = copy.deepcopy(section)
        sec.verbosity(verbosity=False)
        lst = sec.get_str().split('\n')
        string = '\n'.join((line for line in lst if not line.startswith('&')))
        return cls.from_str(string)

    def get_str(self) -> str:
        """Convert model to a GTH-formatted string."""
        if self.info is None or self.n_elecs is None or self.r_loc is None or (self.nexp_ppl is None) or (self.c_exp_ppl is None) or (self.radii is None) or (self.nprj is None) or (self.nprj_ppnl is None) or (self.hprj_ppnl is None):
            raise ValueError('Must initialize all attributes in order to get string')
        out = f'{self.element} {self.name} {' '.join(self.alias_names)}\n'
        out += f'{' '.join((str(self.n_elecs[i]) for i in range(len(self.n_elecs))))}\n'
        out += f'{self.r_loc: .14f} {self.nexp_ppl} '
        for i in range(self.nexp_ppl):
            out += f'{self.c_exp_ppl[i]: .14f} '
        out += '\n'
        out += f'{self.nprj} \n'
        for idx in range(self.nprj):
            total_fill = self.nprj_ppnl[idx] * 20 + 24
            tmp = f'{self.radii[idx]: .14f} {self.nprj_ppnl[idx]: d}'
            out += f'{tmp:>{''}{24}}'
            for i in range(self.nprj_ppnl[idx]):
                k = total_fill - 24 if i == 0 else total_fill
                tmp = ' '.join((f'{v: .14f}' for v in self.hprj_ppnl[idx][i].values()))
                out += f'{tmp:>{''}{k}}'
                out += '\n'
        return out

    @classmethod
    def from_str(cls, string: str) -> Self:
        """Initialize model from a GTH formatted string."""
        lines = [line for line in string.split('\n') if line]
        firstline = lines[0].split()
        element, name, aliases = (firstline[0], firstline[1], firstline[2:])
        info = PotentialInfo.from_str(name).as_dict()
        for alias in aliases:
            for k, v in PotentialInfo.from_str(alias).as_dict().items():
                if info[k] is None:
                    info[k] = v
        info = PotentialInfo.from_dict(info)
        potential: Literal['All Electron', 'Pseudopotential']
        if any(('ALL' in x for x in [name, *aliases])):
            potential = 'All Electron'
            info.electrons = Element(element).Z
        else:
            potential = 'Pseudopotential'
        n_elecs = {idx: int(n_elec) for idx, n_elec in enumerate(lines[1].split())}
        info.electrons = sum(n_elecs.values())
        thirdline = lines[2].split()
        r_loc, nexp_ppl, c_exp_ppl = (float(thirdline[0]), int(thirdline[1]), list(map(float, thirdline[2:])))
        nprj = int(lines[3].split()[0]) if len(lines) > 3 else 0
        radii: dict[int, float] = {}
        nprj_ppnl: dict[int, int] = {}
        hprj_ppnl: dict[int, dict] = {}
        lines = lines[4:]
        i = 0
        ll = 0
        L = 0
        while ll < nprj:
            line = lines[i].split()
            radii[ll] = float(line[0])
            nprj_ppnl[ll] = int(line[1])
            hprj_ppnl[ll] = {x: {} for x in range(nprj_ppnl[ll])}
            _line = [float(i) for i in line[2:]]
            hprj_ppnl[ll][0] = {j: float(ln) for j, ln in enumerate(_line)}
            L = 1
            i += 1
            while nprj_ppnl[ll] > L:
                line2 = list(map(float, lines[i].split()))
                hprj_ppnl[ll][L] = {j: float(ln) for j, ln in enumerate(line2)}
                i += 1
                L += 1
            ll += 1
        return cls(element=Element(element), name=name, alias_names=aliases, potential=potential, n_elecs=n_elecs, r_loc=r_loc, nexp_ppl=nexp_ppl, c_exp_ppl=c_exp_ppl, info=info, radii=radii, nprj=nprj, nprj_ppnl=nprj_ppnl, hprj_ppnl=hprj_ppnl)