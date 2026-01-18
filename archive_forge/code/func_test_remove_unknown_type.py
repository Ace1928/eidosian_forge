from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_remove_unknown_type(self):
    with self.assertRaises(UndefinedTypeCheck) as context:
        TypeChecker().remove('foobar')
    self.assertIn('foobar', str(context.exception))