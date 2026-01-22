import os.path as op
import numpy as np
import networkx as nx
import pickle
from ... import logging
from ..base import (
from .base import have_cv
class NetworkBasedStatistic(LibraryBaseInterface):
    """
    Calculates and outputs the average network given a set of input NetworkX gpickle files

    See Also
    --------
    For documentation of Network-based statistic parameters:
    https://github.com/LTS5/connectomeviewer/blob/master/cviewer/libs/pyconto/groupstatistics/nbs/_nbs.py

    Example
    -------
    >>> import nipype.interfaces.cmtk as cmtk
    >>> nbs = cmtk.NetworkBasedStatistic()
    >>> nbs.inputs.in_group1 = ['subj1.pck', 'subj2.pck'] # doctest: +SKIP
    >>> nbs.inputs.in_group2 = ['pat1.pck', 'pat2.pck'] # doctest: +SKIP
    >>> nbs.run()                 # doctest: +SKIP

    """
    input_spec = NetworkBasedStatisticInputSpec
    output_spec = NetworkBasedStatisticOutputSpec
    _pkg = 'cviewer'

    def _run_interface(self, runtime):
        from cviewer.libs.pyconto.groupstatistics import nbs
        THRESH = self.inputs.threshold
        K = self.inputs.number_of_permutations
        TAIL = self.inputs.t_tail
        edge_key = self.inputs.edge_key
        details = edge_key + '-thresh-' + str(THRESH) + '-k-' + str(K) + '-tail-' + TAIL + '.pck'
        X = ntwks_to_matrices(self.inputs.in_group1, edge_key)
        Y = ntwks_to_matrices(self.inputs.in_group2, edge_key)
        PVAL, ADJ, _ = nbs.compute_nbs(X, Y, THRESH, K, TAIL)
        iflogger.info('p-values:')
        iflogger.info(PVAL)
        pADJ = ADJ.copy()
        for idx, _ in enumerate(PVAL):
            x, y = np.where(ADJ == idx + 1)
            pADJ[x, y] = PVAL[idx]
        nbsgraph = nx.from_numpy_array(ADJ)
        nbs_pval_graph = nx.from_numpy_array(pADJ)
        nbsgraph = nx.relabel_nodes(nbsgraph, lambda x: x + 1)
        nbs_pval_graph = nx.relabel_nodes(nbs_pval_graph, lambda x: x + 1)
        if isdefined(self.inputs.node_position_network):
            node_ntwk_name = self.inputs.node_position_network
        else:
            node_ntwk_name = self.inputs.in_group1[0]
        node_network = _read_pickle(node_ntwk_name)
        iflogger.info('Populating node dictionaries with attributes from %s', node_ntwk_name)
        for nid, ndata in node_network.nodes(data=True):
            nbsgraph.nodes[nid] = ndata
            nbs_pval_graph.nodes[nid] = ndata
        path = op.abspath('NBS_Result_' + details)
        iflogger.info(path)
        with open(path, 'wb') as f:
            pickle.dump(nbsgraph, f, pickle.HIGHEST_PROTOCOL)
        iflogger.info('Saving output NBS edge network as %s', path)
        pval_path = op.abspath('NBS_P_vals_' + details)
        iflogger.info(pval_path)
        with open(pval_path, 'wb') as f:
            pickle.dump(nbs_pval_graph, f, pickle.HIGHEST_PROTOCOL)
        iflogger.info('Saving output p-value network as %s', pval_path)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        THRESH = self.inputs.threshold
        K = self.inputs.number_of_permutations
        TAIL = self.inputs.t_tail
        edge_key = self.inputs.edge_key
        details = edge_key + '-thresh-' + str(THRESH) + '-k-' + str(K) + '-tail-' + TAIL + '.pck'
        path = op.abspath('NBS_Result_' + details)
        pval_path = op.abspath('NBS_P_vals_' + details)
        outputs['nbs_network'] = path
        outputs['nbs_pval_network'] = pval_path
        outputs['network_files'] = [path, pval_path]
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext