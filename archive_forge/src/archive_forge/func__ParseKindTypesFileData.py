from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _ParseKindTypesFileData(self, file_data):
    """Parse Resource Types data into input string Array."""
    if not file_data:
        return None
    return [x for x in re.split('\\s+|,+', file_data) if x]