from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core.util import times
Update core/http_timeout using args and function timeout.

  Args:
    args: The arguments from the command line parser
    function: function definition
    api_version: v1 or v2
    release_track: ALPHA, BETA, or GA
  