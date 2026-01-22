import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def add_title_section(self, title):
    title_section = self.add_new_section('title')
    title_section.style.h1(title)
    return title_section