import codecs
import sys
import click
from celery.app import defaults
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
from celery.utils.functional import pass1
def _compat_key(key, namespace='CELERY'):
    key = key.upper()
    if not key.startswith(namespace):
        key = '_'.join([namespace, key])
    return key