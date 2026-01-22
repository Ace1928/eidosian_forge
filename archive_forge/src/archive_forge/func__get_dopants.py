from __future__ import annotations
import warnings
import numpy as np
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.core import Element, Species
def _get_dopants(substitutions, num_dopants, match_oxi_sign):
    """Utility method to get n- and p-type dopants from a list of substitutions."""
    n_type = [pred for pred in substitutions if pred['dopant_species'].oxi_state > pred['original_species'].oxi_state and (not match_oxi_sign or np.sign(pred['dopant_species'].oxi_state) == np.sign(pred['original_species'].oxi_state))]
    p_type = [pred for pred in substitutions if pred['dopant_species'].oxi_state < pred['original_species'].oxi_state and (not match_oxi_sign or np.sign(pred['dopant_species'].oxi_state) == np.sign(pred['original_species'].oxi_state))]
    return {'n_type': n_type[:num_dopants], 'p_type': p_type[:num_dopants]}