import datetime
import decimal
import functools
import logging
import time
import warnings
from contextlib import contextmanager
from hashlib import md5
from django.apps import apps
from django.db import NotSupportedError
from django.utils.dateparse import parse_time
def truncate_name(identifier, length=None, hash_len=4):
    """
    Shorten an SQL identifier to a repeatable mangled version with the given
    length.

    If a quote stripped name contains a namespace, e.g. USERNAME"."TABLE,
    truncate the table portion only.
    """
    namespace, name = split_identifier(identifier)
    if length is None or len(name) <= length:
        return identifier
    digest = names_digest(name, length=hash_len)
    return '%s%s%s' % ('%s"."' % namespace if namespace else '', name[:length - hash_len], digest)