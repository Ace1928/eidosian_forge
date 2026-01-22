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
class TestRefResolver(TestCase):
    base_uri = ''
    stored_uri = 'foo://stored'
    stored_schema = {'stored': 'schema'}

    def setUp(self):
        self.referrer = {}
        self.store = {self.stored_uri: self.stored_schema}
        self.resolver = validators._RefResolver(self.base_uri, self.referrer, self.store)

    def test_it_does_not_retrieve_schema_urls_from_the_network(self):
        ref = validators.Draft3Validator.META_SCHEMA['id']
        with mock.patch.object(self.resolver, 'resolve_remote') as patched:
            with self.resolver.resolving(ref) as resolved:
                pass
        self.assertEqual(resolved, validators.Draft3Validator.META_SCHEMA)
        self.assertFalse(patched.called)

    def test_it_resolves_local_refs(self):
        ref = '#/properties/foo'
        self.referrer['properties'] = {'foo': object()}
        with self.resolver.resolving(ref) as resolved:
            self.assertEqual(resolved, self.referrer['properties']['foo'])

    def test_it_resolves_local_refs_with_id(self):
        schema = {'id': 'http://bar/schema#', 'a': {'foo': 'bar'}}
        resolver = validators._RefResolver.from_schema(schema, id_of=lambda schema: schema.get('id', ''))
        with resolver.resolving('#/a') as resolved:
            self.assertEqual(resolved, schema['a'])
        with resolver.resolving('http://bar/schema#/a') as resolved:
            self.assertEqual(resolved, schema['a'])

    def test_it_retrieves_stored_refs(self):
        with self.resolver.resolving(self.stored_uri) as resolved:
            self.assertIs(resolved, self.stored_schema)
        self.resolver.store['cached_ref'] = {'foo': 12}
        with self.resolver.resolving('cached_ref#/foo') as resolved:
            self.assertEqual(resolved, 12)

    def test_it_retrieves_unstored_refs_via_requests(self):
        ref = 'http://bar#baz'
        schema = {'baz': 12}
        if 'requests' in sys.modules:
            self.addCleanup(sys.modules.__setitem__, 'requests', sys.modules['requests'])
        sys.modules['requests'] = ReallyFakeRequests({'http://bar': schema})
        with self.resolver.resolving(ref) as resolved:
            self.assertEqual(resolved, 12)

    def test_it_retrieves_unstored_refs_via_urlopen(self):
        ref = 'http://bar#baz'
        schema = {'baz': 12}
        if 'requests' in sys.modules:
            self.addCleanup(sys.modules.__setitem__, 'requests', sys.modules['requests'])
        sys.modules['requests'] = None

        @contextmanager
        def fake_urlopen(url):
            self.assertEqual(url, 'http://bar')
            yield BytesIO(json.dumps(schema).encode('utf8'))
        self.addCleanup(setattr, validators, 'urlopen', validators.urlopen)
        validators.urlopen = fake_urlopen
        with self.resolver.resolving(ref) as resolved:
            pass
        self.assertEqual(resolved, 12)

    def test_it_retrieves_local_refs_via_urlopen(self):
        with tempfile.NamedTemporaryFile(delete=False, mode='wt') as tempf:
            self.addCleanup(os.remove, tempf.name)
            json.dump({'foo': 'bar'}, tempf)
        ref = f'file://{pathname2url(tempf.name)}#foo'
        with self.resolver.resolving(ref) as resolved:
            self.assertEqual(resolved, 'bar')

    def test_it_can_construct_a_base_uri_from_a_schema(self):
        schema = {'id': 'foo'}
        resolver = validators._RefResolver.from_schema(schema, id_of=lambda schema: schema.get('id', ''))
        self.assertEqual(resolver.base_uri, 'foo')
        self.assertEqual(resolver.resolution_scope, 'foo')
        with resolver.resolving('') as resolved:
            self.assertEqual(resolved, schema)
        with resolver.resolving('#') as resolved:
            self.assertEqual(resolved, schema)
        with resolver.resolving('foo') as resolved:
            self.assertEqual(resolved, schema)
        with resolver.resolving('foo#') as resolved:
            self.assertEqual(resolved, schema)

    def test_it_can_construct_a_base_uri_from_a_schema_without_id(self):
        schema = {}
        resolver = validators._RefResolver.from_schema(schema)
        self.assertEqual(resolver.base_uri, '')
        self.assertEqual(resolver.resolution_scope, '')
        with resolver.resolving('') as resolved:
            self.assertEqual(resolved, schema)
        with resolver.resolving('#') as resolved:
            self.assertEqual(resolved, schema)

    def test_custom_uri_scheme_handlers(self):

        def handler(url):
            self.assertEqual(url, ref)
            return schema
        schema = {'foo': 'bar'}
        ref = 'foo://bar'
        resolver = validators._RefResolver('', {}, handlers={'foo': handler})
        with resolver.resolving(ref) as resolved:
            self.assertEqual(resolved, schema)

    def test_cache_remote_on(self):
        response = [object()]

        def handler(url):
            try:
                return response.pop()
            except IndexError:
                self.fail('Response must not have been cached!')
        ref = 'foo://bar'
        resolver = validators._RefResolver('', {}, cache_remote=True, handlers={'foo': handler})
        with resolver.resolving(ref):
            pass
        with resolver.resolving(ref):
            pass

    def test_cache_remote_off(self):
        response = [object()]

        def handler(url):
            try:
                return response.pop()
            except IndexError:
                self.fail('Handler called twice!')
        ref = 'foo://bar'
        resolver = validators._RefResolver('', {}, cache_remote=False, handlers={'foo': handler})
        with resolver.resolving(ref):
            pass

    def test_if_you_give_it_junk_you_get_a_resolution_error(self):
        error = ValueError("Oh no! What's this?")

        def handler(url):
            raise error
        ref = 'foo://bar'
        resolver = validators._RefResolver('', {}, handlers={'foo': handler})
        with self.assertRaises(exceptions._RefResolutionError) as err:
            with resolver.resolving(ref):
                self.fail("Shouldn't get this far!")
        self.assertEqual(err.exception, exceptions._RefResolutionError(error))

    def test_helpful_error_message_on_failed_pop_scope(self):
        resolver = validators._RefResolver('', {})
        resolver.pop_scope()
        with self.assertRaises(exceptions._RefResolutionError) as exc:
            resolver.pop_scope()
        self.assertIn('Failed to pop the scope', str(exc.exception))

    def test_pointer_within_schema_with_different_id(self):
        """
        See #1085.
        """
        schema = validators.Draft7Validator.META_SCHEMA
        one = validators._RefResolver('', schema)
        validator = validators.Draft7Validator(schema, resolver=one)
        self.assertFalse(validator.is_valid({'maxLength': 'foo'}))
        another = {'allOf': [{'$ref': validators.Draft7Validator.META_SCHEMA['$id']}]}
        two = validators._RefResolver('', another)
        validator = validators.Draft7Validator(another, resolver=two)
        self.assertFalse(validator.is_valid({'maxLength': 'foo'}))

    def test_newly_created_validator_with_ref_resolver(self):
        """
        See https://github.com/python-jsonschema/jsonschema/issues/1061#issuecomment-1624266555.
        """

        def handle(uri):
            self.assertEqual(uri, 'http://example.com/foo')
            return {'type': 'integer'}
        resolver = validators._RefResolver('', {}, handlers={'http': handle})
        Validator = validators.create(meta_schema={}, validators=validators.Draft4Validator.VALIDATORS)
        schema = {'$id': 'http://example.com/bar', '$ref': 'foo'}
        validator = Validator(schema, resolver=resolver)
        self.assertEqual((validator.is_valid({}), validator.is_valid(37)), (False, True))

    def test_refresolver_with_pointer_in_schema_with_no_id(self):
        """
        See https://github.com/python-jsonschema/jsonschema/issues/1124#issuecomment-1632574249.
        """
        schema = {'properties': {'x': {'$ref': '#/definitions/x'}}, 'definitions': {'x': {'type': 'integer'}}}
        validator = validators.Draft202012Validator(schema, resolver=validators._RefResolver('', schema))
        self.assertEqual((validator.is_valid({'x': 'y'}), validator.is_valid({'x': 37})), (False, True))