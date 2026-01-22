import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class NumberingConfig(object):
    """Configuration class from elyxer.config file"""
    layouts = {'ordered': ['Chapter', 'Section', 'Subsection', 'Subsubsection', 'Paragraph'], 'roman': ['Part', 'Book']}
    sequence = {'symbols': ['*', '**', '†', '‡', '§', '§§', '¶', '¶¶', '#', '##']}