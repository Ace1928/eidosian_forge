import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
class DocumentClass(object):
    """Details of a LaTeX document class."""

    def __init__(self, document_class, with_part=False):
        self.document_class = document_class
        self._with_part = with_part
        self.sections = ['section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph']
        if self.document_class in ('book', 'memoir', 'report', 'scrbook', 'scrreprt'):
            self.sections.insert(0, 'chapter')
        if self._with_part:
            self.sections.insert(0, 'part')

    def section(self, level):
        """Return the LaTeX section name for section `level`.

        The name depends on the specific document class.
        Level is 1,2,3..., as level 0 is the title.
        """
        if level <= len(self.sections):
            return self.sections[level - 1]
        else:
            return 'DUtitle[section%s]' % roman.toRoman(level)