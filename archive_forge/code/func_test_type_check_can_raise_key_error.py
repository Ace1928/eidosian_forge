from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_type_check_can_raise_key_error(self):
    """
        Make sure no one writes:

            try:
                self._type_checkers[type](...)
            except KeyError:

        ignoring the fact that the function itself can raise that.
        """
    error = KeyError('Stuff')

    def raises_keyerror(checker, instance):
        raise error
    with self.assertRaises(KeyError) as context:
        TypeChecker({'foo': raises_keyerror}).is_type(4, 'foo')
    self.assertIs(context.exception, error)