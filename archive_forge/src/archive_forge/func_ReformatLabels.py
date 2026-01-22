from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def ReformatLabels(args, labels):
    """Reformat label list to encoded labels message.

  Reformatting labels will be done within following two steps,
  1. Filter label strings in a label list.
  2. Convert the filtered label list to OrderedDict.
  3. Encode the OrderedDict format of labels to group.labels message.

  Args:
    args: The argparse namespace.
    labels: list of label strings. e.g.
      ["cloudidentity.googleapis.com/security=",
      "cloudidentity.googleapis.com/groups.discussion_forum"]

  Returns:
    Encoded labels message.

  Raises:
    InvalidArgumentException: If invalid labels string is input.
  """
    filtered_labels = FilterLabels(labels)
    labels_dict = collections.OrderedDict()
    for label in filtered_labels:
        if '=' in label:
            split_label = label.split('=')
            labels_dict[split_label[0]] = split_label[1]
        else:
            labels_dict[label] = ''
    version = GetApiVersion(args)
    messages = ci_client.GetMessages(version)
    return encoding.DictToMessage(labels_dict, messages.Group.LabelsValue)