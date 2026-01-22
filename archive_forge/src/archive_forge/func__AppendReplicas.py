from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def _AppendReplicas(msgs, add_replicas_arg, replica_info_list):
    """Appends each in add_replicas_arg to the given ReplicaInfo list."""
    for replica in add_replicas_arg:
        replica_type = arg_utils.ChoiceToEnum(replica['type'], msgs.ReplicaInfo.TypeValueValuesEnum)
        replica_info_list.append(msgs.ReplicaInfo(location=replica['location'], type=replica_type))