import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
import zlib
from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
def _should_queue(self, link, referrer, rel):
    """
        Determine whether a link URL from a referring page and with a
        particular "rel" attribute should be queued for scraping.
        """
    scheme, netloc, path, _, _, _ = urlparse(link)
    if path.endswith(self.source_extensions + self.binary_extensions + self.excluded_extensions):
        result = False
    elif self.skip_externals and (not link.startswith(self.base_url)):
        result = False
    elif not referrer.startswith(self.base_url):
        result = False
    elif rel not in ('homepage', 'download'):
        result = False
    elif scheme not in ('http', 'https', 'ftp'):
        result = False
    elif self._is_platform_dependent(link):
        result = False
    else:
        host = netloc.split(':', 1)[0]
        if host.lower() == 'localhost':
            result = False
        else:
            result = True
    logger.debug('should_queue: %s (%s) from %s -> %s', link, rel, referrer, result)
    return result