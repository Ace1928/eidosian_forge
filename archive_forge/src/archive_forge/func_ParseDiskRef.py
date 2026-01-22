from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
def ParseDiskRef(self, resources, args, instance_ref):
    if args.disk_scope == 'regional':
        scope = compute_scopes.ScopeEnum.REGION
    else:
        scope = compute_scopes.ScopeEnum.ZONE
    return instance_utils.ParseDiskResource(resources, args.disk, instance_ref.project, instance_ref.zone, scope)