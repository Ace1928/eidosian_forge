import os
import re
import tempfile
import webbrowser
from typing import Any, Callable, Iterable, Tuple, Union
from weakref import WeakKeyDictionary
from twisted.web import http
from w3lib import html
import scrapy
from scrapy.http.response import Response
from scrapy.utils.decorators import deprecated
from scrapy.utils.python import to_bytes, to_unicode
def _remove_html_comments(body):
    start = body.find(b'<!--')
    while start != -1:
        end = body.find(b'-->', start + 1)
        if end == -1:
            return body[:start]
        else:
            body = body[:start] + body[end + 3:]
            start = body.find(b'<!--')
    return body