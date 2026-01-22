import os
import os.path as op
import sys
from datetime import datetime
from copy import deepcopy
import pickle
import shutil
import numpy as np
from ... import config, logging
from ...utils.misc import str2bool
from ...utils.functions import getsource, create_function_from_source
from ...interfaces.base import traits, TraitedSpec, TraitDictObject, TraitListObject
from ...utils.filemanip import save_json
from .utils import (
from .base import EngineBase
from .nodes import MapNode
def _generate_flatgraph(self):
    """Generate a graph containing only Nodes or MapNodes"""
    import networkx as nx
    logger.debug('expanding workflow: %s', self)
    nodes2remove = []
    if not nx.is_directed_acyclic_graph(self._graph):
        raise Exception('Workflow: %s is not a directed acyclic graph (DAG)' % self.name)
    nodes = list(self._graph.nodes)
    for node in nodes:
        logger.debug('processing node: %s', node)
        if isinstance(node, Workflow):
            nodes2remove.append(node)
            for u, _, d in list(self._graph.in_edges(nbunch=node, data=True)):
                logger.debug('in: connections-> %s', str(d['connect']))
                for cd in deepcopy(d['connect']):
                    logger.debug('in: %s', str(cd))
                    dstnode = node.get_node(cd[1].rsplit('.', 1)[0])
                    srcnode = u
                    srcout = cd[0]
                    dstin = cd[1].split('.')[-1]
                    logger.debug('in edges: %s %s %s %s', srcnode, srcout, dstnode, dstin)
                    self.disconnect(u, cd[0], node, cd[1])
                    self.connect(srcnode, srcout, dstnode, dstin)
            for _, v, d in list(self._graph.out_edges(nbunch=node, data=True)):
                logger.debug('out: connections-> %s', str(d['connect']))
                for cd in deepcopy(d['connect']):
                    logger.debug('out: %s', str(cd))
                    dstnode = v
                    if isinstance(cd[0], tuple):
                        parameter = cd[0][0]
                    else:
                        parameter = cd[0]
                    srcnode = node.get_node(parameter.rsplit('.', 1)[0])
                    if isinstance(cd[0], tuple):
                        srcout = list(cd[0])
                        srcout[0] = parameter.split('.')[-1]
                        srcout = tuple(srcout)
                    else:
                        srcout = parameter.split('.')[-1]
                    dstin = cd[1]
                    logger.debug('out edges: %s %s %s %s', srcnode, srcout, dstnode, dstin)
                    self.disconnect(node, cd[0], v, cd[1])
                    self.connect(srcnode, srcout, dstnode, dstin)
            node._generate_flatgraph()
            for innernode in node._graph.nodes():
                innernode._hierarchy = '.'.join((self.name, innernode._hierarchy))
            self._graph.add_nodes_from(node._graph.nodes())
            self._graph.add_edges_from(node._graph.edges(data=True))
    if nodes2remove:
        self._graph.remove_nodes_from(nodes2remove)
    logger.debug('finished expanding workflow: %s', self)