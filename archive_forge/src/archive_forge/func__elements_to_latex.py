import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def _elements_to_latex(elements):
    el_string = ''
    for el in elements:
        num_string = _num_to_latex(el, decimals=decimals)
        el_string += num_string + ' & '
    el_string = el_string[:-2]
    return el_string