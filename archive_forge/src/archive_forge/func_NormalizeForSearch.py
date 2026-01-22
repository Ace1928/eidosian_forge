from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
def NormalizeForSearch(value, html=False):
    """Returns lowercase unicode NFKD form with accents stripped.

  Args:
    value: The value to be normalized.
    html: If True the value is HTML text and HTML tags are converted to spaces.

  Returns:
    The normalized unicode representation of value suitable for cloud search
    matching.
  """
    text = _Stringize(value).lower()
    if html:
        text = re.sub('<[^>]*>', '', text)
    return ''.join([c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)])