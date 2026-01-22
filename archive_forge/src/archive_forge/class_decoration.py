import sys
import os
import re
import warnings
import types
import unicodedata
class decoration(Decorative, Element):

    def get_header(self):
        if not len(self.children) or not isinstance(self.children[0], header):
            self.insert(0, header())
        return self.children[0]

    def get_footer(self):
        if not len(self.children) or not isinstance(self.children[-1], footer):
            self.append(footer())
        return self.children[-1]