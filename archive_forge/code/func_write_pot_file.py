import glob
import os
import re
import sys
from functools import total_ordering
from itertools import dropwhile
from pathlib import Path
import django
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.files.temp import NamedTemporaryFile
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import (
from django.utils.encoding import DEFAULT_LOCALE_ENCODING
from django.utils.functional import cached_property
from django.utils.jslex import prepare_js_for_gettext
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import get_text_list
from django.utils.translation import templatize
def write_pot_file(potfile, msgs):
    """
    Write the `potfile` with the `msgs` contents, making sure its format is
    valid.
    """
    pot_lines = msgs.splitlines()
    if os.path.exists(potfile):
        lines = dropwhile(len, pot_lines)
    else:
        lines = []
        found, header_read = (False, False)
        for line in pot_lines:
            if not found and (not header_read):
                if 'charset=CHARSET' in line:
                    found = True
                    line = line.replace('charset=CHARSET', 'charset=UTF-8')
            if not line and (not found):
                header_read = True
            lines.append(line)
    msgs = '\n'.join(lines)
    with open(potfile, 'a', encoding='utf-8', newline='\n') as fp:
        fp.write(msgs)