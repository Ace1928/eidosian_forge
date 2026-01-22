from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
class UpdateResult(object):
    """Result type for Diff application.

  Attributes:
    needs_update: bool, whether the diff resulted in any changes to the existing
      labels proto.
    _labels: LabelsValue, the new populated LabelsValue object. If needs_update
      is False, this is identical to the original LabelValue object.
  """

    def __init__(self, needs_update, labels):
        self.needs_update = needs_update
        self._labels = labels

    @property
    def labels(self):
        """Returns the new labels.

    Raises:
      ValueError: if needs_update is False.
    """
        if not self.needs_update:
            raise ValueError('If no update is needed (self.needs_update == False), checking labels is unnecessary.')
        return self._labels

    def GetOrNone(self):
        """Returns the new labels if an update is needed or None otherwise.

    NOTE: If this function returns None, make sure not to include the labels
    field in the field mask of the update command. Otherwise, it's possible to
    inadvertently clear the labels on the resource.
    """
        try:
            return self.labels
        except ValueError:
            return None