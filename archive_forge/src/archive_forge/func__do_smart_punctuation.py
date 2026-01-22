import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_smart_punctuation(self, text):
    """Fancifies 'single quotes', "double quotes", and apostrophes.
        Converts --, ---, and ... into en dashes, em dashes, and ellipses.

        Inspiration is: <http://daringfireball.net/projects/smartypants/>
        See "test/tm-cases/smarty_pants.text" for a full discussion of the
        support here and
        <http://code.google.com/p/python-markdown2/issues/detail?id=42> for a
        discussion of some diversion from the original SmartyPants.
        """
    if "'" in text:
        text = self._do_smart_contractions(text)
        text = self._opening_single_quote_re.sub('&#8216;', text)
        text = self._closing_single_quote_re.sub('&#8217;', text)
    if '"' in text:
        text = self._opening_double_quote_re.sub('&#8220;', text)
        text = self._closing_double_quote_re.sub('&#8221;', text)
    text = text.replace('---', '&#8212;')
    text = text.replace('--', '&#8211;')
    text = text.replace('...', '&#8230;')
    text = text.replace(' . . . ', '&#8230;')
    text = text.replace('. . .', '&#8230;')
    if 'footnotes' in self.extras and 'footnote-ref' in text:
        text = text.replace('class="footnote-ref&#8221;', 'class="footnote-ref"')
    return text