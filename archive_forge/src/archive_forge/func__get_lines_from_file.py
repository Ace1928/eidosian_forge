import functools
import inspect
import itertools
import re
import sys
import types
import warnings
from pathlib import Path
from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import PY311, get_docs_version
from django.views.decorators.debug import coroutine_functions_to_sensitive_variables
def _get_lines_from_file(self, filename, lineno, context_lines, loader=None, module_name=None):
    """
        Return context_lines before and after lineno from file.
        Return (pre_context_lineno, pre_context, context_line, post_context).
        """
    source = self._get_source(filename, loader, module_name)
    if source is None:
        return (None, [], None, [])
    if isinstance(source[0], bytes):
        encoding = 'ascii'
        for line in source[:2]:
            match = re.search(b'coding[:=]\\s*([-\\w.]+)', line)
            if match:
                encoding = match[1].decode('ascii')
                break
        source = [str(sline, encoding, 'replace') for sline in source]
    lower_bound = max(0, lineno - context_lines)
    upper_bound = lineno + context_lines
    try:
        pre_context = source[lower_bound:lineno]
        context_line = source[lineno]
        post_context = source[lineno + 1:upper_bound]
    except IndexError:
        return (None, [], None, [])
    return (lower_bound, pre_context, context_line, post_context)