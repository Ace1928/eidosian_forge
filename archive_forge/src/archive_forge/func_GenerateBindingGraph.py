from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def GenerateBindingGraph(bindings: List[base.BindingData], name: str):
    """Produce graph of the given bindings.

  Args:
    bindings: the list of bindings.
    name: name of the graph

  Yields:
    the binding graph in DOT format.
  """
    yield 'strict digraph {graph_name} {{'.format(graph_name=name)
    yield '  node [style="filled" shape=Mrecord penwidth=2]'
    yield '  rankdir=LR'
    yield '\n'
    in_counter = {}
    out_counter = {}
    ids = {}
    for binding in bindings:
        source_id = binding.from_id
        dest_id = binding.to_id
        ids[_NodeName(source_id)] = source_id
        ids[_NodeName(dest_id)] = dest_id
        _CountType(out_counter, source_id)
        _CountType(in_counter, dest_id)
        yield _GraphvizEdge(source_id, dest_id)
    yield '\n'
    for name in ids:
        res_id = ids[name]
        in_count = in_counter.get(res_id.type, 0)
        out_count = out_counter.get(res_id.type, 0)
        yield _GraphvizNode(res_id, in_count, out_count)
    yield '}'