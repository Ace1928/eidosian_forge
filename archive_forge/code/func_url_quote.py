import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def url_quote(s):
    if s is None:
        return ''
    return quote(str(s))