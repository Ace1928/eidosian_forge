from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
import six
from six.moves import range  # pylint: disable=redefined-builtin
def SetLang(self, lang):
    """Sets the ```...``` code block language.

    Args:
      lang: The language name, None if not in a code block, '' is no explicit
        language specified.
    """
    self._lang = lang