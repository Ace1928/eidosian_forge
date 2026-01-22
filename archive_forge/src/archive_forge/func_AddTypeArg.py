from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def AddTypeArg(parser):
    """Adds an integration type arg."""
    parser.add_argument('--type', required=True, help='Type of the integration. To see available types and usage, use "gcloud run integrations types list" command.')