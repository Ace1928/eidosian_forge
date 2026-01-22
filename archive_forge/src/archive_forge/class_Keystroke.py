import re
import time
import platform
from collections import OrderedDict
import six
class Keystroke(six.text_type):
    """
    A unicode-derived class for describing a single keystroke.

    A class instance describes a single keystroke received on input,
    which may contain multiple characters as a multibyte sequence,
    which is indicated by properties :attr:`is_sequence` returning
    ``True``.

    When the string is a known sequence, :attr:`code` matches terminal
    class attributes for comparison, such as ``term.KEY_LEFT``.

    The string-name of the sequence, such as ``u'KEY_LEFT'`` is accessed
    by property :attr:`name`, and is used by the :meth:`__repr__` method
    to display a human-readable form of the Keystroke this class
    instance represents. It may otherwise by joined, split, or evaluated
    just as as any other unicode string.
    """

    def __new__(cls, ucs='', code=None, name=None):
        """Class constructor."""
        new = six.text_type.__new__(cls, ucs)
        new._name = name
        new._code = code
        return new

    @property
    def is_sequence(self):
        """Whether the value represents a multibyte sequence (bool)."""
        return self._code is not None

    def __repr__(self):
        """Docstring overwritten."""
        return six.text_type.__repr__(self) if self._name is None else self._name
    __repr__.__doc__ = six.text_type.__doc__

    @property
    def name(self):
        """String-name of key sequence, such as ``u'KEY_LEFT'`` (str)."""
        return self._name

    @property
    def code(self):
        """Integer keycode value of multibyte sequence (int)."""
        return self._code