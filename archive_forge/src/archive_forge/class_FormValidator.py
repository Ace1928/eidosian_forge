import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class FormValidator(FancyValidator):
    """
    A FormValidator is something that can be chained with a Schema.

    Unlike normal chaining the FormValidator can validate forms that
    aren't entirely valid.

    The important method is .validate(), of course.  It gets passed a
    dictionary of the (processed) values from the form.  If you have
    .validate_partial_form set to True, then it will get the incomplete
    values as well -- check with the "in" operator if the form was able
    to process any particular field.

    Anyway, .validate() should return a string or a dictionary.  If a
    string, it's an error message that applies to the whole form.  If
    not, then it should be a dictionary of fieldName: errorMessage.
    The special key "form" is the error message for the form as a whole
    (i.e., a string is equivalent to {"form": string}).

    Returns None on no errors.
    """
    validate_partial_form = False
    validate_partial_python = None
    validate_partial_other = None

    def is_empty(self, value):
        return False

    def field_is_empty(self, value):
        return is_empty(value)