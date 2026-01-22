from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util as args_labels_util
def SetFunctionLabels(function, update_labels, remove_labels, clear_labels):
    """Set the labels on a function based on args.

  Args:
    function: the function to set the labels on
    update_labels: a dict of <label-name>-<label-value> pairs for the labels to
      be updated, from --update-labels
    remove_labels: a list of the labels to be removed, from --remove-labels
    clear_labels: a bool representing whether or not to clear all labels, from
      --clear-labels

  Returns:
    A bool indicating whether or not any labels were updated on the function.
  """
    labels_to_update = update_labels or {}
    labels_to_update['deployment-tool'] = 'cli-gcloud'
    labels_diff = args_labels_util.Diff(additions=labels_to_update, subtractions=remove_labels, clear=clear_labels)
    messages = api_util.GetApiMessagesModule()
    labels_update = labels_diff.Apply(messages.CloudFunction.LabelsValue, function.labels)
    if labels_update.needs_update:
        function.labels = labels_update.labels
        return True
    return False