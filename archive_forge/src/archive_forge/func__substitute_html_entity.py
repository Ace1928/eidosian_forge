from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
@classmethod
def _substitute_html_entity(cls, matchobj):
    """Used with a regular expression to substitute the
        appropriate HTML entity for a special character string."""
    entity = cls.CHARACTER_TO_HTML_ENTITY.get(matchobj.group(0))
    return '&%s;' % entity