import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class DocumentParameters(object):
    """Global parameters for the document."""
    pdftitle = None
    indentstandard = False
    tocdepth = 10
    startinglevel = 0
    maxdepth = 10
    language = None
    bibliography = None
    outputchanges = False
    displaymode = False