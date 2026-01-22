from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetAdditionalUserLabelsArg(required=False):
    return base.Argument('--additional-user-labels', required=required, default=None, metavar='ADDITIONAL_USER_LABELS', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help='Default user labels to be specified for the job. Keys and values must follow the restrictions specified in https://cloud.google.com/compute/docs/labeling-resources#restrictions.')