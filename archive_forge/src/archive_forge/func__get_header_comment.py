from __future__ import annotations
import datetime
import re
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from copy import copy
from difflib import SequenceMatcher
from email import message_from_string
from heapq import nlargest
from typing import TYPE_CHECKING
from babel import __version__ as VERSION
from babel.core import Locale, UnknownLocaleError
from babel.dates import format_datetime
from babel.messages.plurals import get_plural
from babel.util import LOCALTZ, FixedOffsetTimezone, _cmp, distinct
def _get_header_comment(self) -> str:
    comment = self._header_comment
    year = datetime.datetime.now(LOCALTZ).strftime('%Y')
    if hasattr(self.revision_date, 'strftime'):
        year = self.revision_date.strftime('%Y')
    comment = comment.replace('PROJECT', self.project).replace('VERSION', self.version).replace('YEAR', year).replace('ORGANIZATION', self.copyright_holder)
    locale_name = self.locale.english_name if self.locale else self.locale_identifier
    if locale_name:
        comment = comment.replace('Translations template', f'{locale_name} translations')
    return comment