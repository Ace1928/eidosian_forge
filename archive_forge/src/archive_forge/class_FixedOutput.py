import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FixedOutput(ContainerOutput):
    """Fixed output"""

    def gethtml(self, container):
        """Return constant HTML code"""
        return container.html