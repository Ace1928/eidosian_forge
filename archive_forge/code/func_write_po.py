from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def write_po(fileobj: SupportsWrite[bytes], catalog: Catalog, width: int=76, no_location: bool=False, omit_header: bool=False, sort_output: bool=False, sort_by_file: bool=False, ignore_obsolete: bool=False, include_previous: bool=False, include_lineno: bool=True) -> None:
    """Write a ``gettext`` PO (portable object) template file for a given
    message catalog to the provided file-like object.

    >>> catalog = Catalog()
    >>> catalog.add(u'foo %(name)s', locations=[('main.py', 1)],
    ...             flags=('fuzzy',))
    <Message...>
    >>> catalog.add((u'bar', u'baz'), locations=[('main.py', 3)])
    <Message...>
    >>> from io import BytesIO
    >>> buf = BytesIO()
    >>> write_po(buf, catalog, omit_header=True)
    >>> print(buf.getvalue().decode("utf8"))
    #: main.py:1
    #, fuzzy, python-format
    msgid "foo %(name)s"
    msgstr ""
    <BLANKLINE>
    #: main.py:3
    msgid "bar"
    msgid_plural "baz"
    msgstr[0] ""
    msgstr[1] ""
    <BLANKLINE>
    <BLANKLINE>

    :param fileobj: the file-like object to write to
    :param catalog: the `Catalog` instance
    :param width: the maximum line width for the generated output; use `None`,
                  0, or a negative number to completely disable line wrapping
    :param no_location: do not emit a location comment for every message
    :param omit_header: do not include the ``msgid ""`` entry at the top of the
                        output
    :param sort_output: whether to sort the messages in the output by msgid
    :param sort_by_file: whether to sort the messages in the output by their
                         locations
    :param ignore_obsolete: whether to ignore obsolete messages and not include
                            them in the output; by default they are included as
                            comments
    :param include_previous: include the old msgid as a comment when
                             updating the catalog
    :param include_lineno: include line number in the location comment
    """

    def _normalize(key, prefix=''):
        return normalize(key, prefix=prefix, width=width)

    def _write(text):
        if isinstance(text, str):
            text = text.encode(catalog.charset, 'backslashreplace')
        fileobj.write(text)

    def _write_comment(comment, prefix=''):
        _width = width if width and width > 0 else 76
        for line in wraptext(comment, _width):
            _write(f'#{prefix} {line.strip()}\n')

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
    sort_by = None
    if sort_output:
        sort_by = 'message'
    elif sort_by_file:
        sort_by = 'location'
    for message in _sort_messages(catalog, sort_by=sort_by):
        if not message.id:
            if omit_header:
                continue
            comment_header = catalog.header_comment
            if width and width > 0:
                lines = []
                for line in comment_header.splitlines():
                    lines += wraptext(line, width=width, subsequent_indent='# ')
                comment_header = '\n'.join(lines)
            _write(f'{comment_header}\n')
        for comment in message.user_comments:
            _write_comment(comment)
        for comment in message.auto_comments:
            _write_comment(comment, prefix='.')
        if not no_location:
            locs = []
            try:
                locations = sorted(message.locations, key=lambda x: (x[0], isinstance(x[1], int) and x[1] or -1))
            except TypeError:
                locations = message.locations
            for filename, lineno in locations:
                location = filename.replace(os.sep, '/')
                if lineno and include_lineno:
                    location = f'{location}:{lineno:d}'
                if location not in locs:
                    locs.append(location)
            _write_comment(' '.join(locs), prefix=':')
        if message.flags:
            _write(f'#{', '.join(['', *sorted(message.flags)])}\n')
        if message.previous_id and include_previous:
            _write_comment(f'msgid {_normalize(message.previous_id[0])}', prefix='|')
            if len(message.previous_id) > 1:
                _write_comment('msgid_plural %s' % _normalize(message.previous_id[1]), prefix='|')
        _write_message(message)
        _write('\n')
    if not ignore_obsolete:
        for message in _sort_messages(catalog.obsolete.values(), sort_by=sort_by):
            for comment in message.user_comments:
                _write_comment(comment)
            _write_message(message, prefix='#~ ')
            _write('\n')