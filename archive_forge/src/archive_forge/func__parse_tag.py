from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _parse_tag(self, name):
    """Parse a tag command."""
    from_ = self._get_from(b'tag')
    tagger = self._get_user_info(b'tag', b'tagger', accept_just_who=True)
    message = self._get_data(b'tag', b'message')
    return commands.TagCommand(name, from_, tagger, message)