import re
import string
import warnings
from bleach._vendor.html5lib import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib import (
from bleach._vendor.html5lib.constants import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib.constants import (
from bleach._vendor.html5lib.filters.base import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib._inputstream import (
from bleach._vendor.html5lib.serializer import (
from bleach._vendor.html5lib._tokenizer import (
from bleach._vendor.html5lib._trie import (
class BleachHTMLSerializer(HTMLSerializer):
    """HTMLSerializer that undoes & -> &amp; in attributes and sets
    escape_rcdata to True
    """
    escape_rcdata = True

    def escape_base_amp(self, stoken):
        """Escapes just bare & in HTML attribute values"""
        stoken = stoken.replace('&amp;', '&')
        for part in next_possible_entity(stoken):
            if not part:
                continue
            if part.startswith('&'):
                entity = match_entity(part)
                if entity is not None and convert_entity(entity) is not None:
                    yield f'&{entity};'
                    part = part[len(entity) + 2:]
                    if part:
                        yield part
                    continue
            yield part.replace('&', '&amp;')

    def serialize(self, treewalker, encoding=None):
        """Wrap HTMLSerializer.serialize and conver & to &amp; in attribute values

        Note that this converts & to &amp; in attribute values where the & isn't
        already part of an unambiguous character entity.

        """
        in_tag = False
        after_equals = False
        for stoken in super().serialize(treewalker, encoding):
            if in_tag:
                if stoken == '>':
                    in_tag = False
                elif after_equals:
                    if stoken != '"':
                        yield from self.escape_base_amp(stoken)
                        after_equals = False
                        continue
                elif stoken == '=':
                    after_equals = True
                yield stoken
            else:
                if stoken.startswith('<'):
                    in_tag = True
                yield stoken