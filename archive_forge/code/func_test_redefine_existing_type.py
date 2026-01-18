from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_redefine_existing_type(self):
    self.assertEqual(TypeChecker().redefine('two', object()).redefine('two', equals_2), TypeChecker().redefine('two', equals_2))