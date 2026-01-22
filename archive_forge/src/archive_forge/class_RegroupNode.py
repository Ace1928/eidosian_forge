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
class RegroupNode(Node):

    def __init__(self, target, expression, var_name):
        self.target = target
        self.expression = expression
        self.var_name = var_name

    def resolve_expression(self, obj, context):
        context[self.var_name] = obj
        return self.expression.resolve(context, ignore_failures=True)

    def render(self, context):
        obj_list = self.target.resolve(context, ignore_failures=True)
        if obj_list is None:
            context[self.var_name] = []
            return ''
        context[self.var_name] = [GroupedResult(grouper=key, list=list(val)) for key, val in groupby(obj_list, lambda obj: self.resolve_expression(obj, context))]
        return ''