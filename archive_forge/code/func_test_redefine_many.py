from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_redefine_many(self):
    self.assertEqual(TypeChecker().redefine_many({'foo': int, 'bar': str}), TypeChecker().redefine('foo', int).redefine('bar', str))