import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from itertools import cycle as itertools_cycle
from itertools import groupby
from django.conf import settings
from django.utils import timezone
from django.utils.html import conditional_escape, escape, format_html
from django.utils.lorem_ipsum import paragraphs, words
from django.utils.safestring import mark_safe
from .base import (
from .context import Context
from .defaultfilters import date
from .library import Library
from .smartif import IfParser, Literal
class DebugNode(Node):

    def render(self, context):
        if not settings.DEBUG:
            return ''
        from pprint import pformat
        output = [escape(pformat(val)) for val in context]
        output.append('\n\n')
        output.append(escape(pformat(sys.modules)))
        return ''.join(output)