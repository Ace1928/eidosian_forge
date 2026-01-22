from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def AddServiceUpdateArgs(parser):
    """Adds service arguments for update."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--add-service', help='Name of the Cloud Run service to attach to the integration.')
    group.add_argument('--remove-service', help='Name of the Cloud Run service to remove from the integration.')