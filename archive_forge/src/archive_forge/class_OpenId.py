import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class OpenId(FancyValidator):
    """
    OpenId validator.

    ::
        >>> v = OpenId(add_schema=True)
        >>> v.to_python(' example.net ')
        'http://example.net'
        >>> v.to_python('@TurboGears')
        'xri://@TurboGears'
        >>> w = OpenId(add_schema=False)
        >>> w.to_python(' example.net ')
        Traceback (most recent call last):
        ...
        Invalid: "example.net" is not a valid OpenId (it is neither an URL nor an XRI)
        >>> w.to_python('!!1000')
        '!!1000'
        >>> w.to_python('look@me.com')
        Traceback (most recent call last):
        ...
        Invalid: "look@me.com" is not a valid OpenId (it is neither an URL nor an XRI)

    """
    messages = dict(badId=_('"%(id)s" is not a valid OpenId (it is neither an URL nor an XRI)'))

    def __init__(self, add_schema=False, **kwargs):
        """Create an OpenId validator.

        @param add_schema: Should the schema be added if not present?
        @type add_schema: C{bool}

        """
        self.url_validator = URL(add_http=add_schema)
        self.iname_validator = XRI(add_schema, xri_type='i-name')
        self.inumber_validator = XRI(add_schema, xri_type='i-number')

    def _convert_to_python(self, value, state):
        value = value.strip()
        try:
            return self.url_validator.to_python(value, state)
        except Invalid:
            try:
                return self.iname_validator.to_python(value, state)
            except Invalid:
                try:
                    return self.inumber_validator.to_python(value, state)
                except Invalid:
                    pass
        raise Invalid(self.message('badId', state, id=value), value, state)

    def _validate_python(self, value, state):
        self._convert_to_python(value, state)