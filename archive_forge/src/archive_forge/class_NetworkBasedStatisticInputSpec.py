import os.path as op
import numpy as np
import networkx as nx
import pickle
from ... import logging
from ..base import (
from .base import have_cv
class NetworkBasedStatisticInputSpec(BaseInterfaceInputSpec):
    in_group1 = InputMultiPath(File(exists=True), mandatory=True, desc='Networks for the first group of subjects')
    in_group2 = InputMultiPath(File(exists=True), mandatory=True, desc='Networks for the second group of subjects')
    node_position_network = File(desc='An optional network used to position the nodes for the output networks')
    number_of_permutations = traits.Int(1000, usedefault=True, desc='Number of permutations to perform')
    threshold = traits.Float(3, usedefault=True, desc='T-statistic threshold')
    t_tail = traits.Enum('left', 'right', 'both', usedefault=True, desc='Can be one of "left", "right", or "both"')
    edge_key = traits.Str('number_of_fibers', usedefault=True, desc='Usually "number_of_fibers, "fiber_length_mean", "fiber_length_std" for matrices made with CMTKSometimes "weight" or "value" for functional networks.')
    out_nbs_network = File(desc='Output network with edges identified by the NBS')
    out_nbs_pval_network = File(desc='Output network with p-values to weight the edges identified by the NBS')