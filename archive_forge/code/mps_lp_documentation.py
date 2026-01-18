import re
from . import constants as const

    adapted from Julian Märte (https://github.com/pchtsp/pysmps)
    returns a dictionary with the contents of the model.
    This dictionary can be used to generate an LpProblem

    :param path: path of mps file
    :param sense: 1 for minimize, -1 for maximize
    :param dropConsNames: if True, do not store the names of constraints
    :return: a dictionary with all the problem data
    