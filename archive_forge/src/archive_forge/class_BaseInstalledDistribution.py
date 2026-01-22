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
class BaseInstalledDistribution(Distribution):
    """
    This is the base class for installed distributions (whether PEP 376 or
    legacy).
    """
    hasher = None

    def __init__(self, metadata, path, env=None):
        """
        Initialise an instance.
        :param metadata: An instance of :class:`Metadata` which describes the
                         distribution. This will normally have been initialised
                         from a metadata file in the ``path``.
        :param path:     The path of the ``.dist-info`` or ``.egg-info``
                         directory for the distribution.
        :param env:      This is normally the :class:`DistributionPath`
                         instance where this distribution was found.
        """
        super(BaseInstalledDistribution, self).__init__(metadata)
        self.path = path
        self.dist_path = env

    def get_hash(self, data, hasher=None):
        """
        Get the hash of some data, using a particular hash algorithm, if
        specified.

        :param data: The data to be hashed.
        :type data: bytes
        :param hasher: The name of a hash implementation, supported by hashlib,
                       or ``None``. Examples of valid values are ``'sha1'``,
                       ``'sha224'``, ``'sha384'``, '``sha256'``, ``'md5'`` and
                       ``'sha512'``. If no hasher is specified, the ``hasher``
                       attribute of the :class:`InstalledDistribution` instance
                       is used. If the hasher is determined to be ``None``, MD5
                       is used as the hashing algorithm.
        :returns: The hash of the data. If a hasher was explicitly specified,
                  the returned hash will be prefixed with the specified hasher
                  followed by '='.
        :rtype: str
        """
        if hasher is None:
            hasher = self.hasher
        if hasher is None:
            hasher = hashlib.md5
            prefix = ''
        else:
            hasher = getattr(hashlib, hasher)
            prefix = '%s=' % self.hasher
        digest = hasher(data).digest()
        digest = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
        return '%s%s' % (prefix, digest)