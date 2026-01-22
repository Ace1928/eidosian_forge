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
def FilterLabels(labels):
    """Filter label strings in label list.

  Filter labels (list of strings) with the following conditions,
  1. If 'label' has 'key' and 'value' OR 'key' only, then add the label to
  filtered label list. (e.g. 'label_key=label_value', 'label_key')
  2. If 'label' has an equal sign but no 'value', then add the 'key' to filtered
  label list. (e.g. 'label_key=' ==> 'label_key')
  3. If 'label' has invalid format of string, throw an InvalidArgumentException.
  (e.g. 'label_key=value1=value2')

  Args:
    labels: list of label strings.

  Returns:
    Filtered label list.

  Raises:
    InvalidArgumentException: If invalid labels string is input.
  """
    if not labels:
        raise exceptions.InvalidArgumentException('labels', 'labels can not be an empty string')
    label_list = labels.split(',')
    filtered_labels = []
    for label in label_list:
        if '=' in label:
            split_label = label.split('=')
            if len(split_label) > 2:
                raise exceptions.InvalidArgumentException('labels', 'Invalid format of label string has been input. Label: ' + label)
            if split_label[1]:
                filtered_labels.append(label)
            else:
                filtered_labels.append(split_label[0])
        else:
            filtered_labels.append(label)
    return filtered_labels