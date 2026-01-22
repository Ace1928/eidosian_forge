from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
class BoxLineCharacters(object):
    """Box/line drawing characters.

  The element names are from ISO 8879:1986//ENTITIES Box and Line Drawing//EN:
    http://www.w3.org/2003/entities/iso8879doc/isobox.html
  """