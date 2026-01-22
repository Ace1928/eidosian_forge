import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _details_to_str(details, special=None):
    """Convert a details dict to a string.

    :param details: A dictionary mapping short names to ``Content`` objects.
    :param special: If specified, an attachment that should have special
        attention drawn to it. The primary attachment. Normally it's the
        traceback that caused the test to fail.
    :return: A formatted string that can be included in text test results.
    """
    empty_attachments = []
    binary_attachments = []
    text_attachments = []
    special_content = None
    for key, content in sorted(details.items()):
        if content.content_type.type != 'text':
            binary_attachments.append((key, content.content_type))
            continue
        text = content.as_text().strip()
        if not text:
            empty_attachments.append(key)
            continue
        if key == special:
            special_content = f'{text}\n'
            continue
        text_attachments.append(_format_text_attachment(key, text))
    if text_attachments and (not text_attachments[-1].endswith('\n')):
        text_attachments.append('')
    if special_content:
        text_attachments.append(special_content)
    lines = []
    if binary_attachments:
        lines.append('Binary content:\n')
        for name, content_type in binary_attachments:
            lines.append(f'  {name} ({content_type})\n')
    if empty_attachments:
        lines.append('Empty attachments:\n')
        for name in empty_attachments:
            lines.append(f'  {name}\n')
    if (binary_attachments or empty_attachments) and text_attachments:
        lines.append('\n')
    lines.append('\n'.join(text_attachments))
    return ''.join(lines)