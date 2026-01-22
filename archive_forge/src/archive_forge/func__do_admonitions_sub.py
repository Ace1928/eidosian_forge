import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_admonitions_sub(self, match):
    lead_indent, admonition_name, title, body = match.groups()
    admonition_type = '<strong>%s</strong>' % admonition_name
    if admonition_name.lower() == 'admonition':
        admonition_class = 'admonition'
    else:
        admonition_class = 'admonition %s' % admonition_name.lower()
    if title:
        title = '<em>%s</em>' % title
    body = self._run_block_gamut('\n%s\n' % self._uniform_outdent(body)[1])
    admonition = self._uniform_indent('%s\n%s\n\n%s\n' % (admonition_type, title, body), self.tab, False)
    admonition = '<aside class="%s">\n%s</aside>' % (admonition_class, admonition)
    return self._uniform_indent(admonition, lead_indent, False)