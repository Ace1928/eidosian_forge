from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.command_lib.org_policies import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _GetPolicyMessageName(release_track):
    """Returns the organization policy message name based on the release_track."""
    api_version = org_policy_service.GetApiVersion(release_track).capitalize()
    return 'GoogleCloudOrgpolicy' + api_version + 'Policy'