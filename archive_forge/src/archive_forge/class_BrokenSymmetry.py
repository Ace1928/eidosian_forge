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
class BrokenSymmetry(Section):
    """
    Define the required atomic orbital occupation assigned in initialization
    of the density matrix, by adding or subtracting electrons from specific
    angular momentum channels. It works only with GUESS ATOMIC.
    """

    def __init__(self, l_alpha: Sequence=(-1,), n_alpha: Sequence=(0,), nel_alpha: Sequence=(-1,), l_beta: Sequence=(-1,), n_beta: Sequence=(0,), nel_beta: Sequence=(-1,)):
        """
        Initialize the broken symmetry section.

        Args:
            l_alpha: Angular momentum quantum number of the orbitals whose occupation is changed
            n_alpha: Principal quantum number of the orbitals whose occupation is changed.
                Default is the first not occupied
            nel_alpha: Orbital occupation change per angular momentum quantum number. In
                unrestricted calculations applied to spin alpha
            l_beta: Same as L_alpha for beta channel
            n_beta: Same as N_alpha for beta channel
            nel_beta: Same as NEL_alpha for beta channel
        """
        self.l_alpha = l_alpha
        self.n_alpha = n_alpha
        self.nel_alpha = nel_alpha
        self.l_beta = l_beta
        self.n_beta = n_beta
        self.nel_beta = nel_beta
        description = 'Define the required atomic orbital occupation assigned in initialization of the density matrix, by adding or subtracting electrons from specific angular momentum channels. It works only with GUESS ATOMIC'
        keywords_alpha = {'L': Keyword('L', *map(int, l_alpha)), 'N': Keyword('N', *map(int, n_alpha)), 'NEL': Keyword('NEL', *map(int, nel_alpha))}
        alpha = Section('ALPHA', keywords=keywords_alpha, subsections={}, repeats=False)
        keywords_beta = {'L': Keyword('L', *map(int, l_beta)), 'N': Keyword('N', *map(int, n_beta)), 'NEL': Keyword('NEL', *map(int, nel_beta))}
        beta = Section('BETA', keywords=keywords_beta, subsections={}, repeats=False)
        super().__init__('BS', description=description, subsections={'ALPHA': alpha, 'BETA': beta}, keywords={}, repeats=False)

    @classmethod
    def from_el(cls, el: Element, oxi_state: int=0, spin: int=0) -> Self:
        """Create section from element, oxidation state, and spin."""
        el = el if isinstance(el, Element) else Element(el)

        def f(x):
            return {'s': 0, 'p': 1, 'd': 2, 'f': 4}.get(x)

        def f2(x):
            return {0: 2, 1: 6, 2: 10, 3: 14}.get(x)

        def f3(x):
            return {0: 2, 1: 6, 2: 10, 3: 14}.get(x)
        es = el.electronic_structure
        esv = [(int(_[0]), f(_[1]), int(_[2:])) for _ in es.split('.') if '[' not in _]
        esv.sort(key=lambda x: (x[0], x[1]), reverse=True)
        tmp = oxi_state
        l_alpha = []
        l_beta = []
        nel_alpha = []
        nel_beta = []
        n_alpha = []
        n_beta = []
        unpaired_orbital: tuple[int, int, int] = (0, 0, 0)
        while tmp:
            tmp2 = -min((esv[0][2], tmp)) if tmp > 0 else min((f2(esv[0][1]) - esv[0][2], -tmp))
            l_alpha.append(esv[0][1])
            l_beta.append(esv[0][1])
            nel_alpha.append(tmp2)
            nel_beta.append(tmp2)
            n_alpha.append(esv[0][0])
            n_beta.append(esv[0][0])
            tmp += tmp2
            unpaired_orbital = (esv[0][0], esv[0][1], esv[0][2] + tmp2)
            esv.pop(0)
        if unpaired_orbital is None:
            raise ValueError('unpaired_orbital cannot be None.')
        if spin == 'low-up':
            spin = unpaired_orbital[2] % 2
        elif spin == 'low-down':
            spin = -(unpaired_orbital[2] % 2)
        elif spin == 'high-up':
            spin = unpaired_orbital[2] % (f2(unpaired_orbital[1]) // 2)
        elif spin == 'high-down':
            spin = -(unpaired_orbital[2] % (f2(unpaired_orbital[1]) // 2))
        if spin:
            for i in reversed(range(len(nel_alpha))):
                nel_alpha[i] += min((spin, f3(l_alpha[i]) - oxi_state))
                nel_beta[i] -= min((spin, f3(l_beta[i]) - oxi_state))
                if spin > 0:
                    spin -= min((spin, f3(l_alpha[i]) - oxi_state))
                else:
                    spin += min((spin, f3(l_beta[i]) - oxi_state))
        return BrokenSymmetry(l_alpha=l_alpha, l_beta=l_beta, nel_alpha=nel_alpha, nel_beta=nel_beta, n_beta=n_beta, n_alpha=n_alpha)