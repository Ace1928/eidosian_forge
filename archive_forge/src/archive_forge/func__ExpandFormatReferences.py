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
def _ExpandFormatReferences(self, doc):
    """Expand {...} references in doc."""
    doc = self._ExpandHelpText(doc)
    doc = NormalizeExampleSection(doc)
    pat = re.compile('^ *(\\$ .{%d,})$' % (_SPLIT - _FIRST_INDENT - _SECTION_INDENT), re.M)
    pos = 0
    rep = ''
    while True:
        match = pat.search(doc, pos)
        if not match:
            break
        rep += doc[pos:match.start(1)] + ExampleCommandLineSplitter().Split(doc[match.start(1):match.end(1)])
        pos = match.end(1)
    if rep:
        doc = rep + doc[pos:]
    return doc