import base64
import binascii
import functools
import hashlib
import importlib
import math
import warnings
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.crypto import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.module_loading import import_string
from django.utils.translation import gettext_noop as _
def _check_encode_args(self, password, salt):
    if password is None:
        raise TypeError('password must be provided.')
    if not salt or '$' in salt:
        raise ValueError('salt must be provided and cannot contain $.')