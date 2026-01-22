from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _write_message(message, prefix=''):
    if isinstance(message.id, (list, tuple)):
        if message.context:
            _write(f'{prefix}msgctxt {_normalize(message.context, prefix)}\n')
        _write(f'{prefix}msgid {_normalize(message.id[0], prefix)}\n')
        _write(f'{prefix}msgid_plural {_normalize(message.id[1], prefix)}\n')
        for idx in range(catalog.num_plurals):
            try:
                string = message.string[idx]
            except IndexError:
                string = ''
            _write(f'{prefix}msgstr[{idx:d}] {_normalize(string, prefix)}\n')
    else:
        if message.context:
            _write(f'{prefix}msgctxt {_normalize(message.context, prefix)}\n')
        _write(f'{prefix}msgid {_normalize(message.id, prefix)}\n')
        _write(f'{prefix}msgstr {_normalize(message.string or '', prefix)}\n')