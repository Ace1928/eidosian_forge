import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class ConfirmType(FancyValidator):
    """
    Confirms that the input/output is of the proper type.

    Uses the parameters:

    subclass:
        The class or a tuple of classes; the item must be an instance
        of the class or a subclass.
    type:
        A type or tuple of types (or classes); the item must be of
        the exact class or type.  Subclasses are not allowed.

    Examples::

        >>> cint = ConfirmType(subclass=int)
        >>> cint.to_python(True)
        True
        >>> cint.to_python('1')
        Traceback (most recent call last):
            ...
        Invalid: '1' is not a subclass of <type 'int'>
        >>> cintfloat = ConfirmType(subclass=(float, int))
        >>> cintfloat.to_python(1.0), cintfloat.from_python(1.0)
        (1.0, 1.0)
        >>> cintfloat.to_python(1), cintfloat.from_python(1)
        (1, 1)
        >>> cintfloat.to_python(None)
        Traceback (most recent call last):
            ...
        Invalid: None is not a subclass of one of the types <type 'float'>, <type 'int'>
        >>> cint2 = ConfirmType(type=int)
        >>> cint2(accept_python=False).from_python(True)
        Traceback (most recent call last):
            ...
        Invalid: True must be of the type <type 'int'>
    """
    accept_iterator = True
    subclass = None
    type = None
    messages = dict(subclass=_('%(object)r is not a subclass of %(subclass)s'), inSubclass=_('%(object)r is not a subclass of one of the types %(subclassList)s'), inType=_('%(object)r must be one of the types %(typeList)s'), type=_('%(object)r must be of the type %(type)s'))

    def __init__(self, *args, **kw):
        FancyValidator.__init__(self, *args, **kw)
        if self.subclass:
            if isinstance(self.subclass, list):
                self.subclass = tuple(self.subclass)
            elif not isinstance(self.subclass, tuple):
                self.subclass = (self.subclass,)
            self._validate_python = self.confirm_subclass
        if self.type:
            if isinstance(self.type, list):
                self.type = tuple(self.type)
            elif not isinstance(self.type, tuple):
                self.type = (self.type,)
            self._validate_python = self.confirm_type

    def confirm_subclass(self, value, state):
        if not isinstance(value, self.subclass):
            if len(self.subclass) == 1:
                msg = self.message('subclass', state, object=value, subclass=self.subclass[0])
            else:
                subclass_list = ', '.join(map(str, self.subclass))
                msg = self.message('inSubclass', state, object=value, subclassList=subclass_list)
            raise Invalid(msg, value, state)

    def confirm_type(self, value, state):
        for t in self.type:
            if type(value) is t:
                break
        else:
            if len(self.type) == 1:
                msg = self.message('type', state, object=value, type=self.type[0])
            else:
                msg = self.message('inType', state, object=value, typeList=', '.join(map(str, self.type)))
            raise Invalid(msg, value, state)
        return value

    def is_empty(self, value):
        return False