from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def SetWorkflowLoggingArg(loglevel, workflow, updated_fields):
    """Sets --call-log-level for the workflow based on the arguments.

  Also updates updated_fields accordingly.

  Args:
    loglevel: Parsed callLogLevel to be set on the workflow.
    workflow: The workflow in which to set the call-log-level.
    updated_fields: A list to which the call-log-level field will be added if
      needed.
  """
    if loglevel is not None:
        workflow.callLogLevel = loglevel
        updated_fields.append('callLogLevel')