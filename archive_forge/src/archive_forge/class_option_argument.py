import sys
import os
import re
import warnings
import types
import unicodedata
class option_argument(Part, TextElement):

    def astext(self):
        return self.get('delimiter', ' ') + TextElement.astext(self)