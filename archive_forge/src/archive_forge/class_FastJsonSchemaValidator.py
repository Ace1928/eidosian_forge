from __future__ import annotations
import os
import fastjsonschema
import jsonschema
from fastjsonschema import JsonSchemaException as _JsonSchemaException
from jsonschema import Draft4Validator as _JsonSchemaValidator
from jsonschema.exceptions import ErrorTree, ValidationError
class FastJsonSchemaValidator(JsonSchemaValidator):
    """A schema validator using fastjsonschema."""
    name = 'fastjsonschema'

    def __init__(self, schema):
        """Initialize the validator."""
        super().__init__(schema)
        self._validator = fastjsonschema.compile(schema)

    def validate(self, data):
        """Validate incoming data."""
        try:
            self._validator(data)
        except _JsonSchemaException as error:
            raise ValidationError(str(error), schema_path=error.path) from error

    def iter_errors(self, data, schema=None):
        """Iterate over errors in incoming data."""
        if schema is not None:
            return super().iter_errors(data, schema)
        errors = []
        validate_func = self._validator
        try:
            validate_func(data)
        except _JsonSchemaException as error:
            errors = [ValidationError(str(error), schema_path=error.path)]
        return errors

    def error_tree(self, errors):
        """Create an error tree for the errors."""
        msg = 'JSON schema error introspection not enabled for fastjsonschema'
        raise NotImplementedError(msg)