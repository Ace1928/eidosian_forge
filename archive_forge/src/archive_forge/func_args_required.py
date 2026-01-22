from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
@property
def args_required(self):
    """True if resource is required & has a single anchor with no fallthroughs.

    Returns:
      bool, whether the argument group should be required.
    """
    if self.allow_empty:
        return False
    anchors = self._GetAnchors()
    if len(anchors) != 1:
        return False
    anchor = anchors[0]
    if self.fallthroughs_map.get(anchor.name, []):
        return False
    return True