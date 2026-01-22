import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
class DebControl(DebPart):

    def scripts(self):
        """ Return a dictionary of maintainer scripts (postinst, prerm, ...)
        mapping script names to script text. """
        scripts = {}
        for fname in MAINT_SCRIPTS:
            if self.has_file(fname):
                data = self.get_content(fname)
                if data is not None:
                    scripts[fname] = data
        return scripts

    def debcontrol(self):
        """ Return the debian/control as a Deb822 (a Debian-specific dict-like
        class) object.

        For a string representation of debian/control try
        .get_content('control') """
        return Deb822(self.get_content(CONTROL_FILE))

    @overload
    def md5sums(self, encoding=None, errors=None):
        pass

    @overload
    def md5sums(self, encoding, errors=None):
        pass

    def md5sums(self, encoding=None, errors=None):
        """ Return a dictionary mapping filenames (of the data part) to
        md5sums. Fails if the control part does not contain a 'md5sum' file.

        Keys of the returned dictionary are the left-hand side values of lines
        in the md5sums member of control.tar.gz, usually file names relative to
        the file system root (without heading '/' or './').

        The returned keys are Unicode objects if an encoding is specified,
        otherwise binary. The returned values are always Unicode."""
        if not self.has_file(MD5_FILE):
            raise DebError("'%s' file not found, can't list MD5 sums" % MD5_FILE)
        md5_file = self.get_file(MD5_FILE, encoding=encoding, errors=errors)
        sums = {}
        newline = '\r\n'
        if encoding is None:
            newline = b'\r\n'
        for line in md5_file.readlines():
            md5, fname = line.rstrip(newline).split(None, 1)
            if isinstance(md5, bytes):
                sums[fname] = md5.decode()
            else:
                sums[fname] = md5
        md5_file.close()
        return sums