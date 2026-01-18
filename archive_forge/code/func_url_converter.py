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
def url_converter(self, name, hashed_files, template=None):
    """
        Return the custom URL converter for the given file name.
        """
    if template is None:
        template = self.default_template

    def converter(matchobj):
        """
            Convert the matched URL to a normalized and hashed URL.

            This requires figuring out which files the matched URL resolves
            to and calling the url() method of the storage.
            """
        matches = matchobj.groupdict()
        matched = matches['matched']
        url = matches['url']
        if re.match('^[a-z]+:', url):
            return matched
        if url.startswith('/') and (not url.startswith(settings.STATIC_URL)):
            return matched
        url_path, fragment = urldefrag(url)
        if not url_path:
            return matched
        if url_path.startswith('/'):
            assert url_path.startswith(settings.STATIC_URL)
            target_name = url_path.removeprefix(settings.STATIC_URL)
        else:
            source_name = name if os.sep == '/' else name.replace(os.sep, '/')
            target_name = posixpath.join(posixpath.dirname(source_name), url_path)
        hashed_url = self._url(self._stored_name, unquote(target_name), force=True, hashed_files=hashed_files)
        transformed_url = '/'.join(url_path.split('/')[:-1] + hashed_url.split('/')[-1:])
        if fragment:
            transformed_url += ('?#' if '?#' in url else '#') + fragment
        matches['url'] = unquote(transformed_url)
        return template % matches
    return converter