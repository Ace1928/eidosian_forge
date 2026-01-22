import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
def _build_template(self):
    name = self.name
    exception = ParserException(self.ctx, self.line, 'Deprecated Kivy lang template syntax used "{}". Templates will be removed in a future version'.format(name))
    if name not in ('[FileListEntry@FloatLayout+TreeViewNode]', '[FileIconEntry@Widget]', '[AccordionItemTitle@Label]'):
        Logger.warning(exception)
    if __debug__:
        trace('Builder: build template for %s' % name)
    if name[0] != '[' or name[-1] != ']':
        raise ParserException(self.ctx, self.line, 'Invalid template (must be inside [])')
    item_content = name[1:-1]
    if '@' not in item_content:
        raise ParserException(self.ctx, self.line, 'Invalid template name (missing @)')
    template_name, template_root_cls = item_content.split('@')
    self.ctx.templates.append((template_name, template_root_cls, self))