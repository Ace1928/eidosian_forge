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
def CreateUsingExistingConfig(args, config):
    """Create a new CMMR instance config based on an existing GMMR/CMMR config."""
    msgs = apis.GetMessagesModule('spanner', 'v1')
    display_name = args.display_name if args.display_name else config.displayName
    labels = args.labels if args.labels else config.labels
    base_config = config.baseConfig if config.baseConfig else config.name
    replica_info_list = config.replicas
    if args.skip_replicas:
        _SkipReplicas(msgs, args.skip_replicas, replica_info_list)
    if args.add_replicas:
        _AppendReplicas(msgs, args.add_replicas, replica_info_list)
    return _Create(msgs, args.config, display_name, base_config, replica_info_list, labels, args.validate_only, args.etag)