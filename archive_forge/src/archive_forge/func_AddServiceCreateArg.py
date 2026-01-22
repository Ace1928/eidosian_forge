from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def AddServiceCreateArg(parser):
    """Adds a service arg for create."""
    parser.add_argument('--service', help='Name of the Cloud Run service to attach to the integration. It is required for some integrations.')