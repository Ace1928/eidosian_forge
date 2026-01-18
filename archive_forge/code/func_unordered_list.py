import random as random_module
import re
import types
import warnings
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation, getcontext
from functools import wraps
from inspect import unwrap
from operator import itemgetter
from pprint import pformat
from urllib.parse import quote
from django.utils import formats
from django.utils.dateformat import format, time_format
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.encoding import iri_to_uri
from django.utils.html import avoid_wrapping, conditional_escape, escape, escapejs
from django.utils.html import json_script as _json_script
from django.utils.html import linebreaks, strip_tags
from django.utils.html import urlize as _urlize
from django.utils.safestring import SafeData, mark_safe
from django.utils.text import Truncator, normalize_newlines, phone2numeric
from django.utils.text import slugify as _slugify
from django.utils.text import wrap
from django.utils.timesince import timesince, timeuntil
from django.utils.translation import gettext, ngettext
from .base import VARIABLE_ATTRIBUTE_SEPARATOR
from .library import Library
@register.filter(is_safe=True, needs_autoescape=True)
def unordered_list(value, autoescape=True):
    """
    Recursively take a self-nested list and return an HTML unordered list --
    WITHOUT opening and closing <ul> tags.

    Assume the list is in the proper format. For example, if ``var`` contains:
    ``['States', ['Kansas', ['Lawrence', 'Topeka'], 'Illinois']]``, then
    ``{{ var|unordered_list }}`` returns::

        <li>States
        <ul>
                <li>Kansas
                <ul>
                        <li>Lawrence</li>
                        <li>Topeka</li>
                </ul>
                </li>
                <li>Illinois</li>
        </ul>
        </li>
    """
    if autoescape:
        escaper = conditional_escape
    else:

        def escaper(x):
            return x

    def walk_items(item_list):
        item_iterator = iter(item_list)
        try:
            item = next(item_iterator)
            while True:
                try:
                    next_item = next(item_iterator)
                except StopIteration:
                    yield (item, None)
                    break
                if isinstance(next_item, (list, tuple, types.GeneratorType)):
                    try:
                        iter(next_item)
                    except TypeError:
                        pass
                    else:
                        yield (item, next_item)
                        item = next(item_iterator)
                        continue
                yield (item, None)
                item = next_item
        except StopIteration:
            pass

    def list_formatter(item_list, tabs=1):
        indent = '\t' * tabs
        output = []
        for item, children in walk_items(item_list):
            sublist = ''
            if children:
                sublist = '\n%s<ul>\n%s\n%s</ul>\n%s' % (indent, list_formatter(children, tabs + 1), indent, indent)
            output.append('%s<li>%s%s</li>' % (indent, escaper(item), sublist))
        return '\n'.join(output)
    return mark_safe(list_formatter(value))