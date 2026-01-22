import sys
import os
import re
import warnings
import types
import unicodedata
class FixedTextElement(TextElement):
    """An element which directly contains preformatted text."""

    def __init__(self, rawsource='', text='', *children, **attributes):
        TextElement.__init__(self, rawsource, text, *children, **attributes)
        self.attributes['xml:space'] = 'preserve'