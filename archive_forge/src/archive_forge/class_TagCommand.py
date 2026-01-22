from __future__ import division
import re
import stat
from .helpers import (
class TagCommand(ImportCommand):

    def __init__(self, id, from_, tagger, message):
        ImportCommand.__init__(self, b'tag')
        self.id = id
        self.from_ = from_
        self.tagger = tagger
        self.message = message

    def __bytes__(self):
        if self.from_ is None:
            from_line = b''
        else:
            from_line = b'\nfrom ' + self.from_
        if self.tagger is None:
            tagger_line = b''
        else:
            tagger_line = b'\ntagger ' + format_who_when(self.tagger)
        if self.message is None:
            msg_section = b''
        else:
            msg = self.message
            msg_section = ('\ndata %d\n' % len(msg)).encode('ascii') + msg
        return b'tag ' + self.id + from_line + tagger_line + msg_section