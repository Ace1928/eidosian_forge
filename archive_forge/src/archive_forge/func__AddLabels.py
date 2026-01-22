from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def _AddLabels(self, new_labels):
    new_labels = new_labels.copy()
    new_labels.update(self._additions)
    return new_labels