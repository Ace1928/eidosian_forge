import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def version_identifier(version_info=None):
    """
    Return a version identifier string built from `version_info`, a
    `docutils.VersionInfo` namedtuple instance or compatible tuple. If
    `version_info` is not provided, by default return a version identifier
    string based on `docutils.__version_info__` (i.e. the current Docutils
    version).
    """
    if version_info is None:
        version_info = __version_info__
    if version_info.micro:
        micro = '.%s' % version_info.micro
    else:
        micro = ''
    releaselevel = release_level_abbreviations[version_info.releaselevel]
    if version_info.serial:
        serial = version_info.serial
    else:
        serial = ''
    if version_info.release:
        dev = ''
    else:
        dev = '.dev'
    version = '%s.%s%s%s%s%s' % (version_info.major, version_info.minor, micro, releaselevel, serial, dev)
    return version