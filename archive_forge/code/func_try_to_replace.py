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
def try_to_replace(self, provider, other, problems):
    """
        Attempt to replace one provider with another. This is typically used
        when resolving dependencies from multiple sources, e.g. A requires
        (B >= 1.0) while C requires (B >= 1.1).

        For successful replacement, ``provider`` must meet all the requirements
        which ``other`` fulfills.

        :param provider: The provider we are trying to replace with.
        :param other: The provider we're trying to replace.
        :param problems: If False is returned, this will contain what
                         problems prevented replacement. This is currently
                         a tuple of the literal string 'cantreplace',
                         ``provider``, ``other``  and the set of requirements
                         that ``provider`` couldn't fulfill.
        :return: True if we can replace ``other`` with ``provider``, else
                 False.
        """
    rlist = self.reqts[other]
    unmatched = set()
    for s in rlist:
        matcher = self.get_matcher(s)
        if not matcher.match(provider.version):
            unmatched.add(s)
    if unmatched:
        problems.add(('cantreplace', provider, other, frozenset(unmatched)))
        result = False
    else:
        self.remove_distribution(other)
        del self.reqts[other]
        for s in rlist:
            self.reqts.setdefault(provider, set()).add(s)
        self.add_distribution(provider)
        result = True
    return result