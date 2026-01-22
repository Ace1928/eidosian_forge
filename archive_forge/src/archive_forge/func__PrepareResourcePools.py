from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.ai.persistent_resources import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.ai.persistent_resources import flags
from googlecloudsdk.command_lib.ai.persistent_resources import persistent_resource_util
from googlecloudsdk.command_lib.ai.persistent_resources import validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _PrepareResourcePools(self, args, api_client):
    persistent_resource_config = api_client.ImportResourceMessage(args.config, 'PersistentResource') if args.config else api_client.PersistentResourceMessage()
    validation.ValidateCreateArgs(args, persistent_resource_config, self._version)
    resource_pool_specs = list(args.resource_pool_spec or [])
    persistent_resource_spec = persistent_resource_util.ConstructResourcePools(api_client, persistent_resource_config=persistent_resource_config, resource_pool_specs=resource_pool_specs)
    return persistent_resource_spec