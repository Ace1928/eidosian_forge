from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetParametersArg(required=False):
    return base.Argument('--parameters', required=required, default=None, metavar='PARAMETERS', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help='User defined parameters for the template.')