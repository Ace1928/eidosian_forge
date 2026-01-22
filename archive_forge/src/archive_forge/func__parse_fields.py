import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
def _parse_fields(self):
    in_select = None
    in_textarea = None
    fields = {}
    text = self.text
    for match in self._tag_re.finditer(text):
        end = match.group(1) == b'/'
        tag = match.group(2).lower()
        if tag not in ('input', 'select', 'option', 'textarea', 'button'):
            continue
        if tag == 'select' and end:
            assert in_select, '%r without starting select' % match.group(0)
            in_select = None
            continue
        if tag == 'textarea' and end:
            assert in_textarea, '</textarea> with no <textarea> at %s' % match.start()
            in_textarea[0].value = html_unquote(self.text[in_textarea[1]:match.start()])
            in_textarea = None
            continue
        if end:
            continue
        attrs = _parse_attrs(match.group(3))
        if 'name' in attrs:
            name = attrs.pop('name')
        else:
            name = None
        if tag == 'option':
            in_select.options.append((attrs.get('value'), 'selected' in attrs))
            continue
        if tag == 'input' and attrs.get('type') == 'radio':
            field = fields.get(name)
            if not field:
                field = Radio(self, tag, name, match.start(), **attrs)
                fields.setdefault(name, []).append(field)
            else:
                field = field[0]
                assert isinstance(field, Radio)
            field.options.append((attrs.get('value'), 'checked' in attrs))
            continue
        tag_type = tag
        if tag == 'input':
            tag_type = attrs.get('type', 'text').lower()
        FieldClass = Field.classes.get(tag_type, Field)
        field = FieldClass(self, tag, name, match.start(), **attrs)
        if tag == 'textarea':
            assert not in_textarea, 'Nested textareas: %r and %r' % (in_textarea, match.group(0))
            in_textarea = (field, match.end())
        elif tag == 'select':
            assert not in_select, 'Nested selects: %r and %r' % (in_select, match.group(0))
            in_select = field
        fields.setdefault(name, []).append(field)
    self.fields = fields