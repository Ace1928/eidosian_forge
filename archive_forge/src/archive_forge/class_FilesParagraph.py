import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class FilesParagraph(_RestrictedWrapper):
    """Represents a Files paragraph of a debian/copyright file.

    This kind of paragraph is used to specify the copyright and license for a
    particular set of files in the package.
    """
    _default_re = re.compile('')

    def __init__(self, data, _internal_validate=True, strict=True):
        super(FilesParagraph, self).__init__(data, _internal_validate)
        if _internal_validate:
            if 'Files' not in data:
                raise MachineReadableFormatError('"Files" field required')
            if 'Copyright' not in data:
                _complain('Files paragraph missing Copyright field', strict)
            if 'License' not in data:
                _complain('Files paragraph missing License field', strict)
            if not self.files:
                _complain('Files paragraph has empty Files field', strict)
        self.__cached_files_pat = ('', self._default_re)

    @classmethod
    def create(cls, files, copyright, license):
        """Create a new FilesParagraph from its required parts.

        :param files: The list of file globs.
        :param copyright: The copyright for the files (free-form text).
        :param license: The Licence for the files.
        """
        p = cls(Deb822ParagraphElement.new_empty_paragraph(), _internal_validate=False)
        p.files = files
        p.copyright = copyright
        p.license = license
        return p

    def files_pattern(self):
        """Returns a regular expression equivalent to the Files globs.

        Caches the result until files is set to a different value.

        Raises ValueError if any of the globs are invalid.
        """
        files_str = self['files']
        if self.__cached_files_pat[0] != files_str:
            self.__cached_files_pat = (files_str, globs_to_re(self.files))
        return self.__cached_files_pat[1]

    def matches(self, filename):
        """Returns True iff filename is matched by a glob in Files."""
        pat = self.files_pattern()
        if pat is None:
            return False
        return pat.match(filename) is not None
    files = RestrictedField('Files', from_str=_SpaceSeparated.from_str, to_str=_SpaceSeparated.to_str, allow_none=False)
    copyright = RestrictedField('Copyright', allow_none=False)
    license = RestrictedField('License', from_str=License.from_str, to_str=License.to_str, allow_none=False)
    comment = RestrictedField('Comment')