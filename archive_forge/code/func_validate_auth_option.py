import json
import os
import re
import uuid
from urllib.parse import urlencode
import tornado.auth
import tornado.gen
import tornado.web
from celery.utils.imports import instantiate
from tornado.options import options
from ..views import BaseHandler
from ..views.error import NotFoundErrorHandler
def validate_auth_option(pattern):
    if pattern.count('*') > 1:
        return False
    if '*' in pattern and '|' in pattern:
        return False
    if '*' in pattern.rsplit('@', 1)[-1]:
        return False
    return True