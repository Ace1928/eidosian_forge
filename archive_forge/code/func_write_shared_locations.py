from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def write_shared_locations(self, paths, dry_run=False):
    """
        Write shared location information to the SHARED file in .dist-info.
        :param paths: A dictionary as described in the documentation for
        :meth:`shared_locations`.
        :param dry_run: If True, the action is logged but no file is actually
                        written.
        :return: The path of the file written to.
        """
    shared_path = os.path.join(self.path, 'SHARED')
    logger.info('creating %s', shared_path)
    if dry_run:
        return None
    lines = []
    for key in ('prefix', 'lib', 'headers', 'scripts', 'data'):
        path = paths[key]
        if os.path.isdir(paths[key]):
            lines.append('%s=%s' % (key, path))
    for ns in paths.get('namespace', ()):
        lines.append('namespace=%s' % ns)
    with codecs.open(shared_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return shared_path