from __future__ import annotations
from collections import deque, namedtuple
from contextlib import contextmanager
from decimal import Decimal
from io import BytesIO
from typing import Any
from unittest import TestCase, mock
from urllib.request import pathname2url
import json
import os
import sys
import tempfile
import warnings
from attrs import define, field
from referencing.jsonschema import DRAFT202012
import referencing.exceptions
from jsonschema import (
class TestValidationErrorMessages(TestCase):

    def message_for(self, instance, schema, *args, **kwargs):
        cls = kwargs.pop('cls', validators._LATEST_VERSION)
        cls.check_schema(schema)
        validator = cls(schema, *args, **kwargs)
        errors = list(validator.iter_errors(instance))
        self.assertTrue(errors, msg=f'No errors were raised for {instance!r}')
        self.assertEqual(len(errors), 1, msg=f'Expected exactly one error, found {errors!r}')
        return errors[0].message

    def test_single_type_failure(self):
        message = self.message_for(instance=1, schema={'type': 'string'})
        self.assertEqual(message, "1 is not of type 'string'")

    def test_single_type_list_failure(self):
        message = self.message_for(instance=1, schema={'type': ['string']})
        self.assertEqual(message, "1 is not of type 'string'")

    def test_multiple_type_failure(self):
        types = ('string', 'object')
        message = self.message_for(instance=1, schema={'type': list(types)})
        self.assertEqual(message, "1 is not of type 'string', 'object'")

    def test_object_with_named_type_failure(self):
        schema = {'type': [{'name': 'Foo', 'minimum': 3}]}
        message = self.message_for(instance=1, schema=schema, cls=validators.Draft3Validator)
        self.assertEqual(message, "1 is not of type 'Foo'")

    def test_minimum(self):
        message = self.message_for(instance=1, schema={'minimum': 2})
        self.assertEqual(message, '1 is less than the minimum of 2')

    def test_maximum(self):
        message = self.message_for(instance=1, schema={'maximum': 0})
        self.assertEqual(message, '1 is greater than the maximum of 0')

    def test_dependencies_single_element(self):
        depend, on = ('bar', 'foo')
        schema = {'dependencies': {depend: on}}
        message = self.message_for(instance={'bar': 2}, schema=schema, cls=validators.Draft3Validator)
        self.assertEqual(message, "'foo' is a dependency of 'bar'")

    def test_object_without_title_type_failure_draft3(self):
        type = {'type': [{'minimum': 3}]}
        message = self.message_for(instance=1, schema={'type': [type]}, cls=validators.Draft3Validator)
        self.assertEqual(message, "1 is not of type {'type': [{'minimum': 3}]}")

    def test_dependencies_list_draft3(self):
        depend, on = ('bar', 'foo')
        schema = {'dependencies': {depend: [on]}}
        message = self.message_for(instance={'bar': 2}, schema=schema, cls=validators.Draft3Validator)
        self.assertEqual(message, "'foo' is a dependency of 'bar'")

    def test_dependencies_list_draft7(self):
        depend, on = ('bar', 'foo')
        schema = {'dependencies': {depend: [on]}}
        message = self.message_for(instance={'bar': 2}, schema=schema, cls=validators.Draft7Validator)
        self.assertEqual(message, "'foo' is a dependency of 'bar'")

    def test_additionalItems_single_failure(self):
        message = self.message_for(instance=[2], schema={'items': [], 'additionalItems': False}, cls=validators.Draft3Validator)
        self.assertIn('(2 was unexpected)', message)

    def test_additionalItems_multiple_failures(self):
        message = self.message_for(instance=[1, 2, 3], schema={'items': [], 'additionalItems': False}, cls=validators.Draft3Validator)
        self.assertIn('(1, 2, 3 were unexpected)', message)

    def test_additionalProperties_single_failure(self):
        additional = 'foo'
        schema = {'additionalProperties': False}
        message = self.message_for(instance={additional: 2}, schema=schema)
        self.assertIn("('foo' was unexpected)", message)

    def test_additionalProperties_multiple_failures(self):
        schema = {'additionalProperties': False}
        message = self.message_for(instance=dict.fromkeys(['foo', 'bar']), schema=schema)
        self.assertIn(repr('foo'), message)
        self.assertIn(repr('bar'), message)
        self.assertIn('were unexpected)', message)

    def test_const(self):
        schema = {'const': 12}
        message = self.message_for(instance={'foo': 'bar'}, schema=schema)
        self.assertIn('12 was expected', message)

    def test_contains_draft_6(self):
        schema = {'contains': {'const': 12}}
        message = self.message_for(instance=[2, {}, []], schema=schema, cls=validators.Draft6Validator)
        self.assertEqual(message, 'None of [2, {}, []] are valid under the given schema')

    def test_invalid_format_default_message(self):
        checker = FormatChecker(formats=())
        checker.checks('thing')(lambda value: False)
        schema = {'format': 'thing'}
        message = self.message_for(instance='bla', schema=schema, format_checker=checker)
        self.assertIn(repr('bla'), message)
        self.assertIn(repr('thing'), message)
        self.assertIn('is not a', message)

    def test_additionalProperties_false_patternProperties(self):
        schema = {'type': 'object', 'additionalProperties': False, 'patternProperties': {'^abc$': {'type': 'string'}, '^def$': {'type': 'string'}}}
        message = self.message_for(instance={'zebra': 123}, schema=schema, cls=validators.Draft4Validator)
        self.assertEqual(message, '{} does not match any of the regexes: {}, {}'.format(repr('zebra'), repr('^abc$'), repr('^def$')))
        message = self.message_for(instance={'zebra': 123, 'fish': 456}, schema=schema, cls=validators.Draft4Validator)
        self.assertEqual(message, '{}, {} do not match any of the regexes: {}, {}'.format(repr('fish'), repr('zebra'), repr('^abc$'), repr('^def$')))

    def test_False_schema(self):
        message = self.message_for(instance='something', schema=False)
        self.assertEqual(message, "False schema does not allow 'something'")

    def test_multipleOf(self):
        message = self.message_for(instance=3, schema={'multipleOf': 2})
        self.assertEqual(message, '3 is not a multiple of 2')

    def test_minItems(self):
        message = self.message_for(instance=[], schema={'minItems': 2})
        self.assertEqual(message, '[] is too short')

    def test_maxItems(self):
        message = self.message_for(instance=[1, 2, 3], schema={'maxItems': 2})
        self.assertEqual(message, '[1, 2, 3] is too long')

    def test_minItems_1(self):
        message = self.message_for(instance=[], schema={'minItems': 1})
        self.assertEqual(message, '[] should be non-empty')

    def test_maxItems_0(self):
        message = self.message_for(instance=[1, 2, 3], schema={'maxItems': 0})
        self.assertEqual(message, '[1, 2, 3] is expected to be empty')

    def test_minLength(self):
        message = self.message_for(instance='', schema={'minLength': 2})
        self.assertEqual(message, "'' is too short")

    def test_maxLength(self):
        message = self.message_for(instance='abc', schema={'maxLength': 2})
        self.assertEqual(message, "'abc' is too long")

    def test_minLength_1(self):
        message = self.message_for(instance='', schema={'minLength': 1})
        self.assertEqual(message, "'' should be non-empty")

    def test_maxLength_0(self):
        message = self.message_for(instance='abc', schema={'maxLength': 0})
        self.assertEqual(message, "'abc' is expected to be empty")

    def test_minProperties(self):
        message = self.message_for(instance={}, schema={'minProperties': 2})
        self.assertEqual(message, '{} does not have enough properties')

    def test_maxProperties(self):
        message = self.message_for(instance={'a': {}, 'b': {}, 'c': {}}, schema={'maxProperties': 2})
        self.assertEqual(message, "{'a': {}, 'b': {}, 'c': {}} has too many properties")

    def test_minProperties_1(self):
        message = self.message_for(instance={}, schema={'minProperties': 1})
        self.assertEqual(message, '{} should be non-empty')

    def test_maxProperties_0(self):
        message = self.message_for(instance={1: 2}, schema={'maxProperties': 0})
        self.assertEqual(message, '{1: 2} is expected to be empty')

    def test_prefixItems_with_items(self):
        message = self.message_for(instance=[1, 2, 'foo'], schema={'items': False, 'prefixItems': [{}, {}]})
        self.assertEqual(message, "Expected at most 2 items but found 1 extra: 'foo'")

    def test_prefixItems_with_multiple_extra_items(self):
        message = self.message_for(instance=[1, 2, 'foo', 5], schema={'items': False, 'prefixItems': [{}, {}]})
        self.assertEqual(message, "Expected at most 2 items but found 2 extra: ['foo', 5]")

    def test_pattern(self):
        message = self.message_for(instance='bbb', schema={'pattern': '^a*$'})
        self.assertEqual(message, "'bbb' does not match '^a*$'")

    def test_does_not_contain(self):
        message = self.message_for(instance=[], schema={'contains': {'type': 'string'}})
        self.assertEqual(message, '[] does not contain items matching the given schema')

    def test_contains_too_few(self):
        message = self.message_for(instance=['foo', 1], schema={'contains': {'type': 'string'}, 'minContains': 2})
        self.assertEqual(message, 'Too few items match the given schema (expected at least 2 but only 1 matched)')

    def test_contains_too_few_both_constrained(self):
        message = self.message_for(instance=['foo', 1], schema={'contains': {'type': 'string'}, 'minContains': 2, 'maxContains': 4})
        self.assertEqual(message, 'Too few items match the given schema (expected at least 2 but only 1 matched)')

    def test_contains_too_many(self):
        message = self.message_for(instance=['foo', 'bar', 'baz'], schema={'contains': {'type': 'string'}, 'maxContains': 2})
        self.assertEqual(message, 'Too many items match the given schema (expected at most 2)')

    def test_contains_too_many_both_constrained(self):
        message = self.message_for(instance=['foo'] * 5, schema={'contains': {'type': 'string'}, 'minContains': 2, 'maxContains': 4})
        self.assertEqual(message, 'Too many items match the given schema (expected at most 4)')

    def test_exclusiveMinimum(self):
        message = self.message_for(instance=3, schema={'exclusiveMinimum': 5})
        self.assertEqual(message, '3 is less than or equal to the minimum of 5')

    def test_exclusiveMaximum(self):
        message = self.message_for(instance=3, schema={'exclusiveMaximum': 2})
        self.assertEqual(message, '3 is greater than or equal to the maximum of 2')

    def test_required(self):
        message = self.message_for(instance={}, schema={'required': ['foo']})
        self.assertEqual(message, "'foo' is a required property")

    def test_dependentRequired(self):
        message = self.message_for(instance={'foo': {}}, schema={'dependentRequired': {'foo': ['bar']}})
        self.assertEqual(message, "'bar' is a dependency of 'foo'")

    def test_oneOf_matches_none(self):
        message = self.message_for(instance={}, schema={'oneOf': [False]})
        self.assertEqual(message, '{} is not valid under any of the given schemas')

    def test_oneOf_matches_too_many(self):
        message = self.message_for(instance={}, schema={'oneOf': [True, True]})
        self.assertEqual(message, '{} is valid under each of True, True')

    def test_unevaluated_items(self):
        schema = {'type': 'array', 'unevaluatedItems': False}
        message = self.message_for(instance=['foo', 'bar'], schema=schema)
        self.assertIn(message, "Unevaluated items are not allowed ('foo', 'bar' were unexpected)")

    def test_unevaluated_items_on_invalid_type(self):
        schema = {'type': 'array', 'unevaluatedItems': False}
        message = self.message_for(instance='foo', schema=schema)
        self.assertEqual(message, "'foo' is not of type 'array'")

    def test_unevaluated_properties_invalid_against_subschema(self):
        schema = {'properties': {'foo': {'type': 'string'}}, 'unevaluatedProperties': {'const': 12}}
        message = self.message_for(instance={'foo': 'foo', 'bar': 'bar', 'baz': 12}, schema=schema)
        self.assertEqual(message, "Unevaluated properties are not valid under the given schema ('bar' was unevaluated and invalid)")

    def test_unevaluated_properties_disallowed(self):
        schema = {'type': 'object', 'unevaluatedProperties': False}
        message = self.message_for(instance={'foo': 'foo', 'bar': 'bar'}, schema=schema)
        self.assertEqual(message, "Unevaluated properties are not allowed ('bar', 'foo' were unexpected)")

    def test_unevaluated_properties_on_invalid_type(self):
        schema = {'type': 'object', 'unevaluatedProperties': False}
        message = self.message_for(instance='foo', schema=schema)
        self.assertEqual(message, "'foo' is not of type 'object'")

    def test_single_item(self):
        schema = {'prefixItems': [{}], 'items': False}
        message = self.message_for(instance=['foo', 'bar', 'baz'], schema=schema)
        self.assertEqual(message, "Expected at most 1 item but found 2 extra: ['bar', 'baz']")

    def test_heterogeneous_additionalItems_with_Items(self):
        schema = {'items': [{}], 'additionalItems': False}
        message = self.message_for(instance=['foo', 'bar', 37], schema=schema, cls=validators.Draft7Validator)
        self.assertEqual(message, "Additional items are not allowed ('bar', 37 were unexpected)")

    def test_heterogeneous_items_prefixItems(self):
        schema = {'prefixItems': [{}], 'items': False}
        message = self.message_for(instance=['foo', 'bar', 37], schema=schema)
        self.assertEqual(message, "Expected at most 1 item but found 2 extra: ['bar', 37]")

    def test_heterogeneous_unevaluatedItems_prefixItems(self):
        schema = {'prefixItems': [{}], 'unevaluatedItems': False}
        message = self.message_for(instance=['foo', 'bar', 37], schema=schema)
        self.assertEqual(message, "Unevaluated items are not allowed ('bar', 37 were unexpected)")

    def test_heterogeneous_properties_additionalProperties(self):
        """
        Not valid deserialized JSON, but this should not blow up.
        """
        schema = {'properties': {'foo': {}}, 'additionalProperties': False}
        message = self.message_for(instance={'foo': {}, 'a': 'baz', 37: 12}, schema=schema)
        self.assertEqual(message, "Additional properties are not allowed (37, 'a' were unexpected)")

    def test_heterogeneous_properties_unevaluatedProperties(self):
        """
        Not valid deserialized JSON, but this should not blow up.
        """
        schema = {'properties': {'foo': {}}, 'unevaluatedProperties': False}
        message = self.message_for(instance={'foo': {}, 'a': 'baz', 37: 12}, schema=schema)
        self.assertEqual(message, "Unevaluated properties are not allowed (37, 'a' were unexpected)")