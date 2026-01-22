from inspect import isclass
import pytest
from spacy.errors import ErrorsWithCodes
class Errors(metaclass=ErrorsWithCodes):
    E001 = 'error description'