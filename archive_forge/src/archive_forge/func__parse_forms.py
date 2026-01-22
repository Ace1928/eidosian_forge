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
def _parse_forms(self):
    forms = self._forms_indexed = {}
    form_texts = []
    started = None
    body = self.body
    body = body.decode('utf8', 'xmlcharrefreplace')
    for match in self._tag_re.finditer(body):
        end = match.group(1) == '/'
        tag = match.group(2).lower()
        if tag != 'form':
            continue
        if end:
            assert started, '</form> unexpected at %s' % match.start()
            form_texts.append(body[started:match.end()])
            started = None
        else:
            assert not started, 'Nested form tags at %s' % match.start()
            started = match.start()
    assert not started, 'Dangling form: %r' % body[started:]
    for i, text in enumerate(form_texts):
        form = Form(self, text)
        forms[i] = form
        if form.id:
            forms[form.id] = form