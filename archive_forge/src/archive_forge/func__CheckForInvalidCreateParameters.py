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
def _CheckForInvalidCreateParameters(self, user_provided_params):
    """Raises an exception that lists the parameters that can't be changed."""
    invalid_params = []
    for param in self.type_metadata.parameters:
        if not param.create_allowed and param.name in user_provided_params:
            invalid_params.append(param.name)
    if invalid_params:
        raise exceptions.ArgumentError('The following parameters are not allowed in create command: {}'.format(self._RemoveEncoding(invalid_params)))