import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
@staticmethod
def __normalize_member(fname):
    """ try (not so hard) to obtain a member file name in a form that is
        stored in the .tar.gz, i.e. starting with ./ """
    fname = str(fname).replace('\\', '/')
    if fname.startswith('./'):
        return fname
    if fname.startswith('/'):
        return '.' + fname
    return './' + fname