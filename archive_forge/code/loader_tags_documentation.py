import posixpath
from collections import defaultdict
from django.utils.safestring import mark_safe
from .base import Node, Template, TemplateSyntaxError, TextNode, Variable, token_kwargs
from .library import Library

        Render the specified template and context. Cache the template object
        in render_context to avoid reparsing and loading when used in a for
        loop.
        