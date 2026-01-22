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
def _get_dot(self, prefix=None, hierarchy=None, colored=False, simple_form=True, level=0):
    """Create a dot file with connection info"""
    import networkx as nx
    if prefix is None:
        prefix = '  '
    if hierarchy is None:
        hierarchy = []
    colorset = ['#FFFFC8', '#0000FF', '#B4B4FF', '#E6E6FF', '#FF0000', '#FFB4B4', '#FFE6E6', '#00A300', '#B4FFB4', '#E6FFE6', '#0000FF', '#B4B4FF']
    if level > len(colorset) - 2:
        level = 3
    dotlist = ['%slabel="%s";' % (prefix, self.name)]
    for node in nx.topological_sort(self._graph):
        fullname = '.'.join(hierarchy + [node.fullname])
        nodename = fullname.replace('.', '_')
        if not isinstance(node, Workflow):
            node_class_name = get_print_name(node, simple_form=simple_form)
            if not simple_form:
                node_class_name = '.'.join(node_class_name.split('.')[1:])
            if hasattr(node, 'iterables') and node.iterables:
                dotlist.append('%s[label="%s", shape=box3d,style=filled, color=black, colorscheme=greys7 fillcolor=2];' % (nodename, node_class_name))
            elif colored:
                dotlist.append('%s[label="%s", style=filled, fillcolor="%s"];' % (nodename, node_class_name, colorset[level]))
            else:
                dotlist.append('%s[label="%s"];' % (nodename, node_class_name))
    for node in nx.topological_sort(self._graph):
        if isinstance(node, Workflow):
            fullname = '.'.join(hierarchy + [node.fullname])
            nodename = fullname.replace('.', '_')
            dotlist.append('subgraph cluster_%s {' % nodename)
            if colored:
                dotlist.append(prefix + prefix + 'edge [color="%s"];' % colorset[level + 1])
                dotlist.append(prefix + prefix + 'style=filled;')
                dotlist.append(prefix + prefix + 'fillcolor="%s";' % colorset[level + 2])
            dotlist.append(node._get_dot(prefix=prefix + prefix, hierarchy=hierarchy + [self.name], colored=colored, simple_form=simple_form, level=level + 3))
            dotlist.append('}')
        else:
            for subnode in self._graph.successors(node):
                if node._hierarchy != subnode._hierarchy:
                    continue
                if not isinstance(subnode, Workflow):
                    nodefullname = '.'.join(hierarchy + [node.fullname])
                    subnodefullname = '.'.join(hierarchy + [subnode.fullname])
                    nodename = nodefullname.replace('.', '_')
                    subnodename = subnodefullname.replace('.', '_')
                    for _ in self._graph.get_edge_data(node, subnode)['connect']:
                        dotlist.append('%s -> %s;' % (nodename, subnodename))
                    logger.debug('connection: %s', dotlist[-1])
    for u, v, d in self._graph.edges(data=True):
        uname = '.'.join(hierarchy + [u.fullname])
        vname = '.'.join(hierarchy + [v.fullname])
        for src, dest in d['connect']:
            uname1 = uname
            vname1 = vname
            if isinstance(src, tuple):
                srcname = src[0]
            else:
                srcname = src
            if '.' in srcname:
                uname1 += '.' + '.'.join(srcname.split('.')[:-1])
            if '.' in dest and '@' not in dest:
                if not isinstance(v, Workflow):
                    if 'datasink' not in str(v._interface.__class__).lower():
                        vname1 += '.' + '.'.join(dest.split('.')[:-1])
                else:
                    vname1 += '.' + '.'.join(dest.split('.')[:-1])
            if uname1.split('.')[:-1] != vname1.split('.')[:-1]:
                dotlist.append('%s -> %s;' % (uname1.replace('.', '_'), vname1.replace('.', '_')))
                logger.debug('cross connection: %s', dotlist[-1])
    return ('\n' + prefix).join(dotlist)