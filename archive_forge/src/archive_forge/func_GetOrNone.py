from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
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