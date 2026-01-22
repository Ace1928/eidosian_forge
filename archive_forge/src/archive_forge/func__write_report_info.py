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
def _write_report_info(self, workingdir, name, graph):
    if workingdir is None:
        workingdir = os.getcwd()
    report_dir = op.join(workingdir, name)
    os.makedirs(report_dir, exist_ok=True)
    shutil.copyfile(op.join(op.dirname(__file__), 'report_template.html'), op.join(report_dir, 'index.html'))
    shutil.copyfile(op.join(op.dirname(__file__), '..', '..', 'external', 'd3.js'), op.join(report_dir, 'd3.js'))
    nodes, groups = topological_sort(graph, depth_first=True)
    graph_file = op.join(report_dir, 'graph1.json')
    json_dict = {'nodes': [], 'links': [], 'groups': [], 'maxN': 0}
    for i, node in enumerate(nodes):
        report_file = '%s/_report/report.rst' % node.output_dir().replace(report_dir, '')
        result_file = '%s/result_%s.pklz' % (node.output_dir().replace(report_dir, ''), node.name)
        json_dict['nodes'].append(dict(name='%d_%s' % (i, node.name), report=report_file, result=result_file, group=groups[i]))
    maxN = 0
    for gid in np.unique(groups):
        procs = [i for i, val in enumerate(groups) if val == gid]
        N = len(procs)
        if N > maxN:
            maxN = N
        json_dict['groups'].append(dict(procs=procs, total=N, name='Group_%05d' % gid))
    json_dict['maxN'] = maxN
    for u, v in graph.in_edges():
        json_dict['links'].append(dict(source=nodes.index(u), target=nodes.index(v), value=1))
    save_json(graph_file, json_dict)
    graph_file = op.join(report_dir, 'graph.json')
    num_nodes = len(nodes)
    if num_nodes > 0:
        index_name = np.ceil(np.log10(num_nodes)).astype(int)
    else:
        index_name = 0
    template = '%%0%dd_' % index_name

    def getname(u, i):
        name_parts = u.fullname.split('.')
        return template % i + name_parts[-1]
    json_dict = []
    for i, node in enumerate(nodes):
        imports = []
        for u, v in graph.in_edges(nbunch=node):
            imports.append(getname(u, nodes.index(u)))
        json_dict.append(dict(name=getname(node, i), size=1, group=groups[i], imports=imports))
    save_json(graph_file, json_dict)