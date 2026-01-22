from collections import OrderedDict
import functools
import numpy as np
from qiskit.utils import optionals as _optionals
from qiskit.result import QuasiDistribution, ProbDistribution
from .exceptions import VisualizationError
from .utils import matplotlib_close_if_inline
Generate the data needed for plotting counts.

    Parameters:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex {'001': 130})
        labels (list): The list of bitstring labels for the plot.
        number_to_keep (int): The number of terms to plot and rest
            is made into a single bar called 'rest'.
        kind (str): One of 'counts' or 'distribution`

    Returns:
        tuple: tuple containing:
            (dict): The labels actually used in the plotting.
            (list): List of ndarrays for the bars in each experiment.
            (list): Indices for the locations of the bars for each
                    experiment.
    