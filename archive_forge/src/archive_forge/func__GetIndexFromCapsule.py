from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _GetIndexFromCapsule(capsule):
    """Returns a help doc index line for a capsule line.

  The capsule line is a formal imperative sentence, preceded by optional
  (RELEASE-TRACK) or [TAG] tags, optionally with markdown attributes. The index
  line has no tags, is not capitalized and has no period, period.

  Args:
    capsule: The capsule line to convert to an index line.

  Returns:
    The help doc index line for a capsule line.
  """
    capsule = re.sub('(\\*?[\\[(][A-Z]+[\\])]\\*? +)*', '', capsule)
    match = re.match('([A-Z])([^A-Z].*)', capsule)
    if match:
        capsule = match.group(1).lower() + match.group(2)
    return capsule.rstrip('.')