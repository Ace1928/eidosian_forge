import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class ISODateTimeConverter(FancyValidator):
    """
    Converts fields which contain both date and time, in the
    ISO 8601 standard format YYYY-MM-DDTHH:MM:SS.

    Stores in a datetime.datetime object.

    Examples::

        >>> tim = ISODateTimeConverter()
        >>> tim.to_python('2012-06-25T05:30:25')
        datetime.datetime(2012, 6, 25, 5, 30, 25)
        >>> tim.to_python('1999-12-01T12:00:00')
        datetime.datetime(1999, 12, 1, 12, 0)
        >>> tim.to_python('2012-06-25 05:30:25')
        Traceback (most recent call last):
            ...
        Invalid: The must enter your date & time in the format YYYY-MM-DDTHH:MM:SS

    """
    datetime_module = None
    messages = dict(invalidFormat=_('The must enter your date & time in the format YYYY-MM-DDTHH:MM:SS'))

    def _convert_to_python(self, value, state):
        dt_mod = import_datetime(self.datetime_module)
        datetime_class = dt_mod.datetime
        try:
            datetime = datetime_class.strptime(value, '%Y-%m-%dT%H:%M:%S')
            return datetime
        except ValueError:
            raise Invalid(self.message('invalidFormat', state), value, state)

    def _convert_from_python(self, value, state):
        return value.isoformat('T')