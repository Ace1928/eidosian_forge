import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class RequireIfMissing(FormValidator):
    """
    Require one field based on another field being present or missing.

    This validator is applied to a form, not an individual field (usually
    using a Schema's ``pre_validators`` or ``chained_validators``) and is
    available under both names ``RequireIfMissing`` and ``RequireIfPresent``.

    If you provide a ``missing`` value (a string key name) then
    if that field is missing the field must be entered.
    This gives you an either/or situation.

    If you provide a ``present`` value (another string key name) then
    if that field is present, the required field must also be present.

    ::

        >>> from formencode import validators
        >>> v = validators.RequireIfPresent('phone_type', present='phone')
        >>> v.to_python(dict(phone_type='', phone='510 420  4577'))
        Traceback (most recent call last):
            ...
        Invalid: You must give a value for phone_type
        >>> v.to_python(dict(phone=''))
        {'phone': ''}

    Note that if you have a validator on the optionally-required
    field, you should probably use ``if_missing=None``.  This way you
    won't get an error from the Schema about a missing value.  For example::

        class PhoneInput(Schema):
            phone = PhoneNumber()
            phone_type = String(if_missing=None)
            chained_validators = [RequireIfPresent('phone_type', present='phone')]
    """
    required = None
    missing = None
    present = None
    __unpackargs__ = ('required',)

    def _convert_to_python(self, value_dict, state):
        is_empty = self.field_is_empty
        if is_empty(value_dict.get(self.required)) and (self.missing and is_empty(value_dict.get(self.missing)) or (self.present and (not is_empty(value_dict.get(self.present))))):
            raise Invalid(_('You must give a value for %s') % self.required, value_dict, state, error_dict={self.required: Invalid(self.message('empty', state), value_dict.get(self.required), state)})
        return value_dict