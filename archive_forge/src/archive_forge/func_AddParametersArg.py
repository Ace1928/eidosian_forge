from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def AddParametersArg(parser):
    """Adds a parameters arg."""
    parser.add_argument('--parameters', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, default={}, metavar='PARAMETER=VALUE', help='Comma-separated list of parameter names and values. Names must be one of the parameters shown when describing the integration type. Only simple values can be specified with this flag.')