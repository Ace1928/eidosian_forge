from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_object_extensions_can_handle_custom_validators(self):
    schema = {'type': 'object', 'required': ['x'], 'properties': {'x': {'type': 'integer'}}}
    type_checker = Draft202012Validator.TYPE_CHECKER.redefine('object', is_object_or_named_tuple)

    def coerce_named_tuple(fn):

        def coerced(validator, value, instance, schema):
            if is_namedtuple(instance):
                instance = instance._asdict()
            return fn(validator, value, instance, schema)
        return coerced
    required = coerce_named_tuple(_keywords.required)
    properties = coerce_named_tuple(_keywords.properties)
    CustomValidator = extend(Draft202012Validator, type_checker=type_checker, validators={'required': required, 'properties': properties})
    validator = CustomValidator(schema)
    Point = namedtuple('Point', ['x', 'y'])
    validator.validate(Point(x=4, y=5))
    with self.assertRaises(ValidationError):
        validator.validate(Point(x='not an integer', y=5))
    validator.validate({'x': 4, 'y': 5})
    with self.assertRaises(ValidationError):
        validator.validate({'x': 'not an integer', 'y': 5})