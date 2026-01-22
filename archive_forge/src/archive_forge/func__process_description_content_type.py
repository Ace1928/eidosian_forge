import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_description_content_type(self, value: str) -> str:
    content_types = {'text/plain', 'text/x-rst', 'text/markdown'}
    message = email.message.EmailMessage()
    message['content-type'] = value
    content_type, parameters = (message.get_content_type().lower(), message['content-type'].params)
    if content_type not in content_types or content_type not in value.lower():
        raise self._invalid_metadata(f'{{field}} must be one of {list(content_types)}, not {value!r}')
    charset = parameters.get('charset', 'UTF-8')
    if charset != 'UTF-8':
        raise self._invalid_metadata(f'{{field}} can only specify the UTF-8 charset, not {list(charset)}')
    markdown_variants = {'GFM', 'CommonMark'}
    variant = parameters.get('variant', 'GFM')
    if content_type == 'text/markdown' and variant not in markdown_variants:
        raise self._invalid_metadata(f'valid Markdown variants for {{field}} are {list(markdown_variants)}, not {variant!r}')
    return value