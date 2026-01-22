from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def CreateLocalSsdMessages(args, resources, messages, location=None, scope=None, project=None, use_disk_type_uri=True):
    """Create messages representing local ssds."""
    local_ssds = []
    for local_ssd_disk in getattr(args, 'local_ssd', []) or []:
        local_ssd = _CreateLocalSsdMessage(resources, messages, local_ssd_disk.get('device-name'), local_ssd_disk.get('interface'), local_ssd_disk.get('size'), location, scope, project, use_disk_type_uri)
        local_ssds.append(local_ssd)
    return local_ssds