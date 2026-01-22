from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def DataAttributeBindingGetIamPolicy(attribute_binding_ref):
    """Get Iam Policy request."""
    get_iam_policy_req = dataplex_api.GetMessageModule().DataplexProjectsLocationsDataAttributeBindingsGetIamPolicyRequest(resource=attribute_binding_ref.RelativeName())
    return dataplex_api.GetClientInstance().projects_locations_dataAttributeBindings.GetIamPolicy(get_iam_policy_req)