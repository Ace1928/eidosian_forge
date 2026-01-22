import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _get_discovery_url_choices(self, project_id=None, allow_version_hack=True, min_version=None, max_version=None):
    """Find potential locations for version discovery URLs.

        min_version and max_version are already normalized, so will either be
        None or a tuple.
        """
    url = urllib.parse.urlparse(self.url.rstrip('/'))
    url_parts = url.path.split('/')
    if project_id and url_parts[-1] == project_id:
        self._saved_project_id = url_parts.pop()
    elif not project_id:
        try:
            normalize_version_number(url_parts[-2])
            self._saved_project_id = url_parts.pop()
        except (IndexError, TypeError):
            pass
    catalog_discovery = versioned_discovery = None
    try:
        url_version = normalize_version_number(url_parts[-1])
        versioned_discovery = urllib.parse.ParseResult(url.scheme, url.netloc, '/'.join(url_parts), url.params, url.query, url.fragment).geturl()
    except TypeError:
        pass
    else:
        is_between = min_version and max_version and version_between(min_version, max_version, url_version)
        exact_match = is_between and max_version and (max_version[0] == url_version[0])
        high_match = is_between and max_version and (max_version[1] != LATEST) and version_match(max_version, url_version)
        if exact_match or is_between:
            self._catalog_matches_version = True
            self._catalog_matches_exactly = exact_match
            catalog_discovery = urllib.parse.ParseResult(url.scheme, url.netloc, '/'.join(url_parts), url.params, url.query, url.fragment).geturl().rstrip('/') + '/'
        if catalog_discovery and (high_match or exact_match):
            yield catalog_discovery
            catalog_discovery = None
        url_parts.pop()
    if allow_version_hack:
        hacked_url = urllib.parse.ParseResult(url.scheme, url.netloc, '/'.join(url_parts), url.params, url.query, url.fragment).geturl()
        if hacked_url != self.catalog_url:
            hacked_url = hacked_url.strip('/') + '/'
        yield hacked_url
        if catalog_discovery:
            yield catalog_discovery
        yield self._get_catalog_discover_hack()
    elif versioned_discovery and self._saved_project_id:
        yield versioned_discovery
    yield self.catalog_url