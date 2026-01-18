import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def translate_words(self, words):
    return [self.translation_map.get(w, w) for w in words]