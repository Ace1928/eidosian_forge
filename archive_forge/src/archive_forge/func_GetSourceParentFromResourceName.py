from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def GetSourceParentFromResourceName(resource_name):
    resource_pattern = re.compile('(organizations|projects|folders)/.*/sources/[0-9]+')
    if not resource_pattern.match(resource_name):
        raise InvalidSCCInputError('When providing a full resource path, it must also include the organization, project, or folder prefix.')
    list_source_components = resource_name.split('/')
    return GetParentFromResourceName(resource_name) + '/' + list_source_components[2] + '/' + list_source_components[3]