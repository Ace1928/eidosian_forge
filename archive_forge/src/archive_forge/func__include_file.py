import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _include_file(context, uri, calling_uri, **kwargs):
    """locate the template from the given uri and include it in
    the current output."""
    template = _lookup_template(context, uri, calling_uri)
    callable_, ctx = _populate_self_namespace(context._clean_inheritance_tokens(), template)
    kwargs = _kwargs_for_include(callable_, context._data, **kwargs)
    if template.include_error_handler:
        try:
            callable_(ctx, **kwargs)
        except Exception:
            result = template.include_error_handler(ctx, compat.exception_as())
            if not result:
                raise
    else:
        callable_(ctx, **kwargs)