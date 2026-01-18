import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp

    Calculates and outputs the average network given a set of input NetworkX gpickle files

    This interface will only keep an edge in the averaged network if that edge is present in
    at least half of the input networks.

    Example
    -------
    >>> import nipype.interfaces.cmtk as cmtk
    >>> avg = cmtk.AverageNetworks()
    >>> avg.inputs.in_files = ['subj1.pck', 'subj2.pck']
    >>> avg.run()                 # doctest: +SKIP

    