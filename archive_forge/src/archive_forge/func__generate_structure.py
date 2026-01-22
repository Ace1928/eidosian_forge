import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def _generate_structure(self, section_names):
    for section_name in section_names:
        self.add_new_section(section_name)