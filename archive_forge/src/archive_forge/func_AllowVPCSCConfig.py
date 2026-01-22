from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import resources
def AllowVPCSCConfig(project_id, location_id):
    """Allows requests in Remote Repository inside VPC SC perimeter."""
    client = GetClient()
    messages = GetMessages()
    vc = messages.VPCSCConfig(name='projects/' + project_id + '/locations/' + location_id + '/vpcscConfig', vpcscPolicy=messages.VPCSCConfig.VpcscPolicyValueValuesEnum.ALLOW)
    update_vpcsc_req = messages.ArtifactregistryProjectsLocationsUpdateVpcscConfigRequest(name='projects/' + project_id + '/locations/' + location_id + '/vpcscConfig', vPCSCConfig=vc)
    return client.projects_locations.UpdateVpcscConfig(update_vpcsc_req)