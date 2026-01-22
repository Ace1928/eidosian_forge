from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
class TransformedPDEntry(PDEntry):
    """
    This class represents a TransformedPDEntry, which allows for a PDEntry to be
    transformed to a different composition coordinate space. It is used in the
    construction of phase diagrams that do not have elements as the terminal
    compositions.
    """
    amount_tol = 1e-05

    def __init__(self, entry, sp_mapping, name=None):
        """
        Args:
            entry (PDEntry): Original entry to be transformed.
            sp_mapping ({Composition: DummySpecies}): dictionary mapping Terminal Compositions to Dummy Species.
        """
        super().__init__(entry.composition, entry.energy, name or entry.name, getattr(entry, 'attribute', None))
        self.original_entry = entry
        self.sp_mapping = sp_mapping
        self.rxn = Reaction(list(self.sp_mapping), [self._composition])
        self.rxn.normalize_to(self.original_entry.composition)
        if not all((self.rxn.get_coeff(comp) <= TransformedPDEntry.amount_tol for comp in self.sp_mapping)):
            raise TransformedPDEntryError('Only reactions with positive amounts of reactants allowed')

    @property
    def composition(self) -> Composition:
        """The composition in the dummy species space.

        Returns:
            Composition
        """
        factor = self._composition.num_atoms / self.original_entry.composition.num_atoms
        trans_comp = {self.sp_mapping[comp]: -self.rxn.get_coeff(comp) for comp in self.sp_mapping}
        trans_comp = {k: v * factor for k, v in trans_comp.items() if v > TransformedPDEntry.amount_tol}
        return Composition(trans_comp)

    def __repr__(self):
        output = [f'TransformedPDEntry {self.composition}', f' with original composition {self.original_entry.composition}', f', energy = {self.original_entry.energy:.4f}']
        return ''.join(output)

    def as_dict(self):
        """
        Returns:
            MSONable dictionary representation of TransformedPDEntry.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'sp_mapping': self.sp_mapping, **self.original_entry.as_dict()}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): dictionary representation of TransformedPDEntry.

        Returns:
            TransformedPDEntry
        """
        sp_mapping = dct['sp_mapping']
        del dct['sp_mapping']
        entry = MontyDecoder().process_decoded(dct)
        return cls(entry, sp_mapping)