import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
class tag_token(token):
    """ Represents a token that is actually a tag.  Currently this is just
    the <img> tag, which takes up visible space just like a word but
    is only represented in a document by a tag.  """

    def __new__(cls, tag, data, html_repr, pre_tags=None, post_tags=None, trailing_whitespace=''):
        obj = token.__new__(cls, '%s: %s' % (type, data), pre_tags=pre_tags, post_tags=post_tags, trailing_whitespace=trailing_whitespace)
        obj.tag = tag
        obj.data = data
        obj.html_repr = html_repr
        return obj

    def __repr__(self):
        return 'tag_token(%s, %s, html_repr=%s, post_tags=%r, pre_tags=%r, trailing_whitespace=%r)' % (self.tag, self.data, self.html_repr, self.pre_tags, self.post_tags, self.trailing_whitespace)

    def html(self):
        return self.html_repr