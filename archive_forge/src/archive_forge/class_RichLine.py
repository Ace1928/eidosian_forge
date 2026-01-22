import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
class RichLine:
    """Rich single-line text.

  Attributes:
    text: A plain string, the raw text represented by this object.  Should not
      contain newlines.
    font_attr_segs: A list of (start, end, font attribute) triples, representing
      richness information applied to substrings of text.
  """

    def __init__(self, text='', font_attr=None):
        """Construct a RichLine with no rich attributes or a single attribute.

    Args:
      text: Raw text string
      font_attr: If specified, a single font attribute to be applied to the
        entire text.  Extending this object via concatenation allows creation
        of text with varying attributes.
    """
        self.text = text
        if font_attr:
            self.font_attr_segs = [(0, len(text), font_attr)]
        else:
            self.font_attr_segs = []

    def __add__(self, other):
        """Concatenate two chunks of maybe rich text to make a longer rich line.

    Does not modify self.

    Args:
      other: Another piece of text to concatenate with this one.
        If it is a plain str, it will be appended to this string with no
        attributes.  If it is a RichLine, it will be appended to this string
        with its attributes preserved.

    Returns:
      A new RichLine comprising both chunks of text, with appropriate
        attributes applied to the corresponding substrings.
    """
        ret = RichLine()
        if isinstance(other, str):
            ret.text = self.text + other
            ret.font_attr_segs = self.font_attr_segs[:]
            return ret
        elif isinstance(other, RichLine):
            ret.text = self.text + other.text
            ret.font_attr_segs = self.font_attr_segs[:]
            old_len = len(self.text)
            for start, end, font_attr in other.font_attr_segs:
                ret.font_attr_segs.append((old_len + start, old_len + end, font_attr))
            return ret
        else:
            raise TypeError('%r cannot be concatenated with a RichLine' % other)

    def __len__(self):
        return len(self.text)