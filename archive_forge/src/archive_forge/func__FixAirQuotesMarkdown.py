from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _FixAirQuotesMarkdown(self, doc):
    """Change ``.*[[:alnum:]]{2,}.*'' quotes => _UserInput(*) in doc."""
    pat = re.compile("[^`](``([^`']*)'')")
    pos = 0
    rep = ''
    for match in pat.finditer(doc):
        if re.search('\\w\\w', match.group(2)):
            quoted_string = self._UserInput(match.group(2))
        else:
            quoted_string = match.group(1)
        rep += doc[pos:match.start(1)] + quoted_string
        pos = match.end(1)
    if rep:
        doc = rep + doc[pos:]
    return doc