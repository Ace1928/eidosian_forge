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
@staticmethod
def FromPrefix(prefix):
    """Gets a ReleaseTrack from the given release track prefix.

    Args:
      prefix: str, The prefix string that might be a release track name.

    Returns:
      ReleaseTrack, The corresponding object or None if the prefix was not a
      valid release track.
    """
    for track in ReleaseTrack:
        if track.prefix == prefix:
            return track
    return None