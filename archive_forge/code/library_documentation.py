from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError

        Render the specified template and context. Cache the template object
        in render_context to avoid reparsing and loading when used in a for
        loop.
        