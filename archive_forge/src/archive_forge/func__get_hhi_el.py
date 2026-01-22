from __future__ import annotations
import os
from monty.design_patterns import singleton
from pymatgen.core import Composition, Element
def _get_hhi_el(self, el_or_symbol):
    """Returns the tuple of HHI_production, HHI reserve for a single element only."""
    if isinstance(el_or_symbol, Element):
        el_or_symbol = el_or_symbol.symbol
    return (self.symbol_hhip_hhir[el_or_symbol][0], self.symbol_hhip_hhir[el_or_symbol][1])