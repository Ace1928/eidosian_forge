from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.console.style import mappings
from googlecloudsdk.core.console.style import text
import six
@classmethod
def FromTextAttributes(cls, text_attributes):
    if not text_attributes:
        return cls(None, [])
    return cls(text_attributes.color, text_attributes.attrs or [])