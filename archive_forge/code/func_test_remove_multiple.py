from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_remove_multiple(self):
    self.assertEqual(TypeChecker({'foo': int, 'bar': str}).remove('foo', 'bar'), TypeChecker())