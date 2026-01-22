from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.core.console import console_io
Prompts to enable the API.

  Args:
    project (str): The project that the API is not enabled on.
    service_token (str): The service token of the API to prompt for.
    enable_by_default (bool): The default choice for the enablement prompt.

  Returns:
    bool, whether or not the API was attempted to be enabled
  