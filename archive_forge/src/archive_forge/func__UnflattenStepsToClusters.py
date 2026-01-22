from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def _UnflattenStepsToClusters(steps):
    """Extract a hierarchy from the steps provided.

  The `step graph' is constructed as follows:

    1. Every node has a `name'. This is flat, something like "s1", "s100".
    2. Each node can depend on others. These edges are specified by "name".
    3. Each node can also have a user_name, like "Foo/Bar". This name creates
       a hierarchy of subgraphs (eg., Foo/Bar and Foo/Baz are in the same
       cluster).

  Args:
    steps: A list of steps from the Job message.
  Returns:
    A Cluster representing the root of the step hierarchy.
  """
    root = _Cluster(None, '')
    for step in steps:
        step_path = _SplitStep(step['properties'].get('user_name', step['name']))
        node = root
        for piece in step_path:
            node = node.GetOrAddChild(piece)
        node.SetStep(step)
    return root