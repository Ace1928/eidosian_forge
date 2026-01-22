from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_client as client
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
import six
def _node_pool_id(self, args: parser_extensions.Namespace):
    """Parses node pool from args and returns its ID."""
    node_pool_ref = self._node_pool_ref(args)
    if node_pool_ref:
        return node_pool_ref.Name()