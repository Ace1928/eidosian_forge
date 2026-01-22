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
def EncodeToBytes(data):
    """Encode data to bytes.

  The primary use case is for base64/mime style 7-bit ascii encoding where the
  encoder input must be bytes. "safe" means that the conversion always returns
  bytes and will not raise codec exceptions.

  If data is text then an 8-bit ascii encoding is attempted, then the console
  encoding, and finally utf-8.

  Args:
    data: Any bytes, string, or object that has str() or unicode() methods.

  Returns:
    A bytes string representation of the data.
  """
    if data is None:
        return b''
    if isinstance(data, bytes):
        return data
    s = six.text_type(data)
    try:
        return s.encode('iso-8859-1')
    except UnicodeEncodeError:
        pass
    try:
        return s.encode(GetConsoleAttr().GetEncoding())
    except UnicodeEncodeError:
        pass
    return s.encode('utf-8')