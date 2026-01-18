from itertools import groupby
from warnings import warn
import numpy as np
from scipy.sparse import find, coo_matrix
def warn_extraneous(extraneous):
    """Display a warning for extraneous keyword arguments.

    The initializer of each solver class is expected to collect keyword
    arguments that it doesn't understand and warn about them. This function
    prints a warning for each key in the supplied dictionary.

    Parameters
    ----------
    extraneous : dict
        Extraneous keyword arguments
    """
    if extraneous:
        warn('The following arguments have no effect for a chosen solver: {}.'.format(', '.join((f'`{x}`' for x in extraneous))), stacklevel=3)