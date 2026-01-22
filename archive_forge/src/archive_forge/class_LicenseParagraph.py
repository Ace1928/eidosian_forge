import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class LicenseParagraph(_RestrictedWrapper):
    """Represents a standalone license paragraph of a debian/copyright file.

    Minimally, this kind of paragraph requires a 'License' field and has no
    'Files' field.  It is used to give a short name to a license text, which
    can be referred to from the header or files paragraphs.
    """

    def __init__(self, data, _internal_validate=True):
        super(LicenseParagraph, self).__init__(data, _internal_validate)
        if _internal_validate:
            if 'License' not in data:
                raise MachineReadableFormatError('"License" field required')
            if 'Files' in data:
                raise MachineReadableFormatError('input appears to be a Files paragraph')

    @classmethod
    def create(cls, license):
        """Returns a LicenseParagraph with the given license."""
        if not isinstance(license, License):
            raise TypeError('license must be a License instance')
        paragraph = cls(Deb822ParagraphElement.new_empty_paragraph(), _internal_validate=False)
        paragraph.license = license
        return paragraph
    license = RestrictedField('License', from_str=License.from_str, to_str=License.to_str, allow_none=False)
    comment = RestrictedField('Comment')
    __files = RestrictedField('Files')