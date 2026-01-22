from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
@classmethod
def GetTrackedAttribute(cls, obj, attribute):
    """Gets the attribute value from obj for tracks.

    The values are checked in ReleaseTrack order.

    Args:
      obj: The object to extract attribute from.
      attribute: The attribute name in object.

    Returns:
      The attribute value from obj for tracks.
    """
    for track in ReleaseTrack:
        if track not in cls._valid_release_tracks:
            continue
        names = []
        names.append(attribute + '_' + track.id)
        if track.prefix:
            names.append(attribute + '_' + track.prefix)
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return getattr(obj, attribute, None)