import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
class href_token(token):
    """ Represents the href in an anchor tag.  Unlike other words, we only
    show the href when it changes.  """
    hide_when_equal = True

    def html(self):
        return ' Link: %s' % self