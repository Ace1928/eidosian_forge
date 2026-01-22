import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class FieldStorageUploadConverter(FancyValidator):
    """
    Handles cgi.FieldStorage instances that are file uploads.

    This doesn't do any conversion, but it can detect empty upload
    fields (which appear like normal fields, but have no filename when
    no upload was given).
    """

    def _convert_to_python(self, value, state=None):
        if isinstance(value, cgi.FieldStorage):
            if getattr(value, 'filename', None):
                return value
            raise Invalid('invalid', value, state)
        else:
            return value

    def is_empty(self, value):
        if isinstance(value, cgi.FieldStorage):
            return not bool(getattr(value, 'filename', None))
        return FancyValidator.is_empty(self, value)