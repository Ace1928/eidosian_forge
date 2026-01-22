import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
class _lazified(object):

    def __init__(self, func, value):
        self.func = func
        self.value = value

    def __str__(self):
        return self.func(self.value)