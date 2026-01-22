import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class StartAppendix(BlackBox):
    """Mark to start an appendix here."""
    'From this point on, all chapters become appendices.'

    def process(self):
        """Activate the special numbering scheme for appendices, using letters."""
        NumberGenerator.generator.startappendix()