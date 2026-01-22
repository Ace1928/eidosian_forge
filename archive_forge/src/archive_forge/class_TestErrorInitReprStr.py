from unittest import TestCase
import textwrap
from jsonschema import exceptions
from jsonschema.validators import _LATEST_VERSION
class TestErrorInitReprStr(TestCase):

    def make_error(self, **kwargs):
        defaults = dict(message='hello', validator='type', validator_value='string', instance=5, schema={'type': 'string'})
        defaults.update(kwargs)
        return exceptions.ValidationError(**defaults)

    def assertShows(self, expected, **kwargs):
        expected = textwrap.dedent(expected).rstrip('\n')
        error = self.make_error(**kwargs)
        message_line, _, rest = str(error).partition('\n')
        self.assertEqual(message_line, error.message)
        self.assertEqual(rest, expected)

    def test_it_calls_super_and_sets_args(self):
        error = self.make_error()
        self.assertGreater(len(error.args), 1)

    def test_repr(self):
        self.assertEqual(repr(exceptions.ValidationError(message='Hello!')), "<ValidationError: 'Hello!'>")

    def test_unset_error(self):
        error = exceptions.ValidationError('message')
        self.assertEqual(str(error), 'message')
        kwargs = {'validator': 'type', 'validator_value': 'string', 'instance': 5, 'schema': {'type': 'string'}}
        for attr in kwargs:
            k = dict(kwargs)
            del k[attr]
            error = exceptions.ValidationError('message', **k)
            self.assertEqual(str(error), 'message')

    def test_empty_paths(self):
        self.assertShows("\n            Failed validating 'type' in schema:\n                {'type': 'string'}\n\n            On instance:\n                5\n            ", path=[], schema_path=[])

    def test_one_item_paths(self):
        self.assertShows("\n            Failed validating 'type' in schema:\n                {'type': 'string'}\n\n            On instance[0]:\n                5\n            ", path=[0], schema_path=['items'])

    def test_multiple_item_paths(self):
        self.assertShows("\n            Failed validating 'type' in schema['items'][0]:\n                {'type': 'string'}\n\n            On instance[0]['a']:\n                5\n            ", path=[0, 'a'], schema_path=['items', 0, 1])

    def test_uses_pprint(self):
        self.assertShows("\n            Failed validating 'maxLength' in schema:\n                {0: 0,\n                 1: 1,\n                 2: 2,\n                 3: 3,\n                 4: 4,\n                 5: 5,\n                 6: 6,\n                 7: 7,\n                 8: 8,\n                 9: 9,\n                 10: 10,\n                 11: 11,\n                 12: 12,\n                 13: 13,\n                 14: 14,\n                 15: 15,\n                 16: 16,\n                 17: 17,\n                 18: 18,\n                 19: 19}\n\n            On instance:\n                [0,\n                 1,\n                 2,\n                 3,\n                 4,\n                 5,\n                 6,\n                 7,\n                 8,\n                 9,\n                 10,\n                 11,\n                 12,\n                 13,\n                 14,\n                 15,\n                 16,\n                 17,\n                 18,\n                 19,\n                 20,\n                 21,\n                 22,\n                 23,\n                 24]\n            ", instance=list(range(25)), schema=dict(zip(range(20), range(20))), validator='maxLength')

    def test_str_works_with_instances_having_overriden_eq_operator(self):
        """
        Check for #164 which rendered exceptions unusable when a
        `ValidationError` involved instances with an `__eq__` method
        that returned truthy values.
        """

        class DontEQMeBro:

            def __eq__(this, other):
                self.fail("Don't!")

            def __ne__(this, other):
                self.fail("Don't!")
        instance = DontEQMeBro()
        error = exceptions.ValidationError('a message', validator='foo', instance=instance, validator_value='some', schema='schema')
        self.assertIn(repr(instance), str(error))