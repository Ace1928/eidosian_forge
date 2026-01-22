import gettext
import os
import re
import textwrap
import warnings
from . import declarative
class FancyValidator(Validator):
    """
    FancyValidator is the (abstract) superclass for various validators
    and converters.  A subclass can validate, convert, or do both.
    There is no formal distinction made here.

    Validators have two important external methods:

    ``.to_python(value, state)``:
      Attempts to convert the value.  If there is a problem, or the
      value is not valid, an Invalid exception is raised.  The
      argument for this exception is the (potentially HTML-formatted)
      error message to give the user.

    ``.from_python(value, state)``:
      Reverses ``.to_python()``.

    These two external methods make use of the following four
    important internal methods that can be overridden.  However,
    none of these *have* to be overridden, only the ones that
    are appropriate for the validator.

    ``._convert_to_python(value, state)``:
      This method converts the source to a Python value.  It returns
      the converted value, or raises an Invalid exception if the
      conversion cannot be done.  The argument to this exception
      should be the error message.  Contrary to ``.to_python()`` it is
      only meant to convert the value, not to fully validate it.

    ``._convert_from_python(value, state)``:
      Should undo ``._convert_to_python()`` in some reasonable way, returning
      a string.

    ``._validate_other(value, state)``:
      Validates the source, before ``._convert_to_python()``, or after
      ``._convert_from_python()``.  It's usually more convenient to use
      ``._validate_python()`` however.

    ``._validate_python(value, state)``:
      Validates a Python value, either the result of ``._convert_to_python()``,
      or the input to ``._convert_from_python()``.

    You should make sure that all possible validation errors are
    raised in at least one these four methods, not matter which.

    Subclasses can also override the ``__init__()`` method
    if the ``declarative.Declarative`` model doesn't work for this.

    Validators should have no internal state besides the
    values given at instantiation.  They should be reusable and
    reentrant.

    All subclasses can take the arguments/instance variables:

    ``if_empty``:
      If set, then this value will be returned if the input evaluates
      to false (empty list, empty string, None, etc), but not the 0 or
      False objects.  This only applies to ``.to_python()``.

    ``not_empty``:
      If true, then if an empty value is given raise an error.
      (Both with ``.to_python()`` and also ``.from_python()``
      if ``._validate_python`` is true).

    ``strip``:
      If true and the input is a string, strip it (occurs before empty
      tests).

    ``if_invalid``:
      If set, then when this validator would raise Invalid during
      ``.to_python()``, instead return this value.

    ``if_invalid_python``:
      If set, when the Python value (converted with
      ``.from_python()``) is invalid, this value will be returned.

    ``accept_python``:
      If True (the default), then ``._validate_python()`` and
      ``._validate_other()`` will not be called when
      ``.from_python()`` is used.

    These parameters are handled at the level of the external
    methods ``.to_python()`` and ``.from_python`` already;
    if you overwrite one of the internal methods, you usually
    don't need to care about them.

    """
    if_invalid = NoDefault
    if_invalid_python = NoDefault
    if_empty = NoDefault
    not_empty = False
    accept_python = True
    strip = False
    messages = dict(empty=_('Please enter a value'), badType=_('The input must be a string (not a %(type)s: %(value)r)'), noneType=_('The input must be a string (not None)'))
    _inheritance_level = 0
    _deprecated_methods = (('_to_python', '_convert_to_python'), ('_from_python', '_convert_from_python'), ('validate_python', '_validate_python'), ('validate_other', '_validate_other'))

    @staticmethod
    def __classinit__(cls, new_attrs):
        Validator.__classinit__(cls, new_attrs)
        cls._inheritance_level += 1
        if '_deprecated_methods' in new_attrs:
            cls._deprecated_methods = cls._deprecated_methods + new_attrs['_deprecated_methods']
        for old, new in cls._deprecated_methods:
            if old in new_attrs:
                if new not in new_attrs:
                    deprecation_warning(old, new, stacklevel=cls._inheritance_level + 2)
                    setattr(cls, new, new_attrs[old])
            elif new in new_attrs:
                setattr(cls, old, deprecated(old=old, new=new)(new_attrs[new]))

    def to_python(self, value, state=None):
        try:
            if self.strip and isinstance(value, str):
                value = value.strip()
            elif hasattr(value, 'mixed'):
                value = value.mixed()
            if self.is_empty(value):
                if self.not_empty:
                    raise Invalid(self.message('empty', state), value, state)
                if self.if_empty is not NoDefault:
                    return self.if_empty
                return self.empty_value(value)
            vo = self._validate_other
            if vo and vo is not self._validate_noop:
                vo(value, state)
            tp = self._convert_to_python
            if tp:
                value = tp(value, state)
            vp = self._validate_python
            if vp and vp is not self._validate_noop:
                vp(value, state)
        except Invalid:
            value = self.if_invalid
            if value is NoDefault:
                raise
        return value

    def from_python(self, value, state=None):
        try:
            if self.strip and isinstance(value, str):
                value = value.strip()
            if not self.accept_python:
                if self.is_empty(value):
                    if self.not_empty:
                        raise Invalid(self.message('empty', state), value, state)
                    return self.empty_value(value)
                vp = self._validate_python
                if vp and vp is not self._validate_noop:
                    vp(value, state)
                fp = self._convert_from_python
                if fp:
                    value = fp(value, state)
                vo = self._validate_other
                if vo and vo is not self._validate_noop:
                    vo(value, state)
            else:
                if self.is_empty(value):
                    return self.empty_value(value)
                fp = self._convert_from_python
                if fp:
                    value = fp(value, state)
        except Invalid:
            value = self.if_invalid_python
            if value is NoDefault:
                raise
        return value

    def is_empty(self, value):
        return is_empty(value)

    def empty_value(self, value):
        return None

    def assert_string(self, value, state):
        if not isinstance(value, str):
            raise Invalid(self.message('badType', state, type=type(value), value=value), value, state)

    def base64encode(self, value):
        """
        Encode a string in base64, stripping whitespace and removing
        newlines.
        """
        return value.encode('base64').strip().replace('\n', '')

    def _validate_noop(self, value, state=None):
        """
        A validation method that doesn't do anything.
        """
        pass
    _validate_python = _validate_other = _validate_noop
    _convert_to_python = _convert_from_python = None