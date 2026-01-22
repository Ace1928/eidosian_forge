from urllib.parse import parse_qsl, unquote, urlparse, urlunparse
from django import template
from django.contrib.admin.utils import quote
from django.urls import Resolver404, get_script_prefix, resolve
from django.utils.http import urlencode
@register.filter
def admin_urlquote(value):
    return quote(value)