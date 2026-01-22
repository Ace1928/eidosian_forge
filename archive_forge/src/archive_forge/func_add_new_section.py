import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def add_new_section(self, name, context=None):
    """Adds a new section to the current document structure

        This document structure will be considered a section to the
        current document structure but will in itself be an entirely
        new document structure that can be written to and have sections
        as well

        :param name: The name of the section.
        :param context: A dictionary of data to store with the strucuture. These
            are only stored per section not the entire structure.
        :rtype: DocumentStructure
        :returns: A new document structure to add to but lives as a section
            to the document structure it was instantiated from.
        """
    section = self.__class__(name=name, target=self.target, context=context)
    section.path = self.path + [name]
    section.style.indentation = self.style.indentation
    section.translation_map = self.translation_map
    section.hrefs = self.hrefs
    self._structure[name] = section
    return section