from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
Returns a list of label protos based on the current state plus edits.

  Args:
    labels: The current label values.
    labels_proto: The LabelEntry proto message class.
    update_labels: A dict of label key=value edits.
    remove_labels: A list of labels keys to remove.

  Returns:
    A list of label protos representing the update and remove edits.
  