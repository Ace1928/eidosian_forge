from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def GetIntegrationValidator(integration_type: str):
    """Gets the integration validator based on the integration type."""
    type_metadata = types_utils.GetTypeMetadata(integration_type)
    if type_metadata is None:
        raise ValueError('Integration type: [{}] has not been defined in types_utils'.format(integration_type))
    return Validator(type_metadata)