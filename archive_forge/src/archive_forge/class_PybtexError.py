from __future__ import absolute_import, unicode_literals
import sys
import six
class PybtexError(Exception):

    def __init__(self, message, filename=None):
        super(PybtexError, self).__init__(message)
        self.filename = filename

    def get_context(self):
        """Return extra error context info."""
        return None

    def get_filename(self):
        """Return filename, if relevant."""
        if self.filename is None or isinstance(self.filename, six.text_type):
            return self.filename
        else:
            from .io import _decode_filename
            return _decode_filename(self.filename, errors='replace')

    def __eq__(self, other):
        return six.text_type(self) == six.text_type(other)

    def __hash__(self):
        return hash(six.text_type(self))