import inspect
import logging
import re
from enum import Enum
from django.template.context import BaseContext
from django.utils.formats import localize
from django.utils.html import conditional_escape, escape
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import get_text_list, smart_split, unescape_string_literal
from django.utils.timezone import template_localtime
from django.utils.translation import gettext_lazy, pgettext_lazy
from .exceptions import TemplateSyntaxError
class TextNode(Node):
    child_nodelists = ()

    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.s[:25])

    def render(self, context):
        return self.s

    def render_annotated(self, context):
        """
        Return the given value.

        The default implementation of this method handles exceptions raised
        during rendering, which is not necessary for text nodes.
        """
        return self.s