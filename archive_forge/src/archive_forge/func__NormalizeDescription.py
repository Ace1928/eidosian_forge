from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _NormalizeDescription(description):
    """Normalizes description text.

  Args:
    description: str, The text to be normalized.

  Returns:
    str, The normalized text.
  """
    if callable(description):
        description = description()
    if description:
        description = textwrap.dedent(description)
    return six.text_type(description or '')