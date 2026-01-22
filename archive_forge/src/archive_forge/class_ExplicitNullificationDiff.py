from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
class ExplicitNullificationDiff(Diff):
    """A change to labels for resources where API requires explicit nullification.

  That is, to clear a label {'foo': 'bar'}, you must pass {'foo': None} to the
  API.
  """

    def _RemoveLabels(self, existing_labels, new_labels):
        """Remove labels."""
        new_labels = new_labels.copy()
        for key in self._subtractions:
            if key in existing_labels:
                new_labels[key] = None
            elif key in new_labels:
                del new_labels[key]
        return new_labels

    def _ClearLabels(self, existing_labels):
        return {key: None for key in existing_labels}