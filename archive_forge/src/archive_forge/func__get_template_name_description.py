from django.template import TemplateSyntaxError
from django.utils.safestring import mark_safe
from django import VERSION as DJANGO_VERSION
from sentry_sdk import _functools, Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
def _get_template_name_description(template_name):
    if isinstance(template_name, (list, tuple)):
        if template_name:
            return '[{}, ...]'.format(template_name[0])
    else:
        return template_name