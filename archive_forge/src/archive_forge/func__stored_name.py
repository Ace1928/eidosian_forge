import json
import os
import posixpath
import re
from hashlib import md5
from urllib.parse import unquote, urldefrag, urlsplit, urlunsplit
from django.conf import STATICFILES_STORAGE_ALIAS, settings
from django.contrib.staticfiles.utils import check_settings, matches_patterns
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, storages
from django.utils.functional import LazyObject
def _stored_name(self, name, hashed_files):
    name = posixpath.normpath(name)
    cleaned_name = self.clean_name(name)
    hash_key = self.hash_key(cleaned_name)
    cache_name = hashed_files.get(hash_key)
    if cache_name is None:
        cache_name = self.clean_name(self.hashed_name(name))
    return cache_name