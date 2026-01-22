import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class CreditCardExpires(FormValidator):
    """
    Checks that credit card expiration date is valid relative to
    the current date.

    You pass in the name of the field that has the credit card
    expiration month and the field with the credit card expiration
    year.

    ::

        >>> ed = CreditCardExpires()
        >>> sorted(ed.to_python({'ccExpiresMonth': '11', 'ccExpiresYear': '2250'}).items())
        [('ccExpiresMonth', '11'), ('ccExpiresYear', '2250')]
        >>> ed.to_python({'ccExpiresMonth': '10', 'ccExpiresYear': '2005'})
        Traceback (most recent call last):
            ...
        Invalid: ccExpiresMonth: Invalid Expiration Date<br>
        ccExpiresYear: Invalid Expiration Date
    """
    validate_partial_form = True
    cc_expires_month_field = 'ccExpiresMonth'
    cc_expires_year_field = 'ccExpiresYear'
    __unpackargs__ = ('cc_expires_month_field', 'cc_expires_year_field')
    datetime_module = None
    messages = dict(notANumber=_('Please enter numbers only for month and year'), invalidNumber=_('Invalid Expiration Date'))

    def validate_partial(self, field_dict, state):
        if not field_dict.get(self.cc_expires_month_field, None) or not field_dict.get(self.cc_expires_year_field, None):
            return None
        self._validate_python(field_dict, state)

    def _validate_python(self, field_dict, state):
        errors = self._validateReturn(field_dict, state)
        if errors:
            raise Invalid('<br>\n'.join(('%s: %s' % (name, value) for name, value in sorted(errors.items()))), field_dict, state, error_dict=errors)

    def _validateReturn(self, field_dict, state):
        ccExpiresMonth = str(field_dict[self.cc_expires_month_field]).strip()
        ccExpiresYear = str(field_dict[self.cc_expires_year_field]).strip()
        try:
            ccExpiresMonth = int(ccExpiresMonth)
            ccExpiresYear = int(ccExpiresYear)
            dt_mod = import_datetime(self.datetime_module)
            now = datetime_now(dt_mod)
            today = datetime_makedate(dt_mod, now.year, now.month, now.day)
            next_month = ccExpiresMonth % 12 + 1
            next_month_year = ccExpiresYear
            if next_month == 1:
                next_month_year += 1
            expires_date = datetime_makedate(dt_mod, next_month_year, next_month, 1)
            assert expires_date > today
        except ValueError:
            return {self.cc_expires_month_field: self.message('notANumber', state), self.cc_expires_year_field: self.message('notANumber', state)}
        except AssertionError:
            return {self.cc_expires_month_field: self.message('invalidNumber', state), self.cc_expires_year_field: self.message('invalidNumber', state)}