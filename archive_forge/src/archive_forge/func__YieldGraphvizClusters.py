from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def _YieldGraphvizClusters(cluster, parent=None):
    if cluster.IsLeaf():
        step = cluster.GetStep()
        yield _NODE_FORMAT.format(name=_EscapeGraphvizId(step['name']), full_name=_EscapeGraphvizId(cluster.Name()), user_name=_EscapeGraphvizId(cluster.Name(relative_to=parent)))
    elif cluster.IsSingleton() or cluster.IsRoot():
        for unused_key, subcluster in cluster.Children():
            for line in _YieldGraphvizClusters(subcluster, parent=parent):
                yield line
    else:
        full_name = cluster.Name()
        yield 'subgraph {0} {{'.format(_EscapeGraphvizId('cluster ' + full_name))
        yield 'style=filled;'
        yield 'bgcolor=white;'
        yield 'labeljust=left;'
        yield 'tooltip={0};'.format(_EscapeGraphvizId(full_name))
        yield 'label={0};'.format(_EscapeGraphvizId(cluster.Name(parent)))
        for unused_key, subgroup in cluster.Children():
            for line in _YieldGraphvizClusters(subgroup, parent=cluster):
                yield line
        yield '}'