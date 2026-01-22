import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
class ConsumerCRUDTests(OAuth1Tests):

    def _consumer_create(self, description=None, description_flag=True, **kwargs):
        if description_flag:
            ref = {'description': description}
        else:
            ref = {}
        if kwargs:
            ref.update(kwargs)
        resp = self.post(self.CONSUMER_URL, body={'consumer': ref})
        consumer = resp.result['consumer']
        consumer_id = consumer['id']
        self.assertEqual(description, consumer['description'])
        self.assertIsNotNone(consumer_id)
        self.assertIsNotNone(consumer['secret'])
        return consumer

    def test_consumer_create(self):
        description = uuid.uuid4().hex
        self._consumer_create(description=description)

    def test_consumer_create_none_desc_1(self):
        self._consumer_create()

    def test_consumer_create_none_desc_2(self):
        self._consumer_create(description_flag=False)

    def test_consumer_create_normalize_field(self):
        field_name = 'some:weird-field'
        field_value = uuid.uuid4().hex
        extra_fields = {field_name: field_value}
        consumer = self._consumer_create(**extra_fields)
        normalized_field_name = 'some_weird_field'
        self.assertEqual(field_value, consumer[normalized_field_name])

    def test_consumer_delete(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        resp = self.delete(self.CONSUMER_URL + '/%s' % consumer_id)
        self.assertResponseStatus(resp, http.client.NO_CONTENT)

    def test_consumer_get_head(self):
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        url = self.CONSUMER_URL + '/%s' % consumer_id
        resp = self.get(url)
        self_url = ['http://localhost/v3', self.CONSUMER_URL, '/', consumer_id]
        self_url = ''.join(self_url)
        self.assertEqual(self_url, resp.result['consumer']['links']['self'])
        self.assertEqual(consumer_id, resp.result['consumer']['id'])
        self.head(url, expected_status=http.client.OK)

    def test_consumer_list(self):
        self._consumer_create()
        resp = self.get(self.CONSUMER_URL)
        entities = resp.result['consumers']
        self.assertIsNotNone(entities)
        self_url = ['http://localhost/v3', self.CONSUMER_URL]
        self_url = ''.join(self_url)
        self.assertEqual(self_url, resp.result['links']['self'])
        self.assertValidListLinks(resp.result['links'])
        self.head(self.CONSUMER_URL, expected_status=http.client.OK)

    def test_consumer_update(self):
        consumer = self._create_single_consumer()
        original_id = consumer['id']
        original_description = consumer['description']
        update_description = original_description + '_new'
        update_ref = {'description': update_description}
        update_resp = self.patch(self.CONSUMER_URL + '/%s' % original_id, body={'consumer': update_ref})
        consumer = update_resp.result['consumer']
        self.assertEqual(update_description, consumer['description'])
        self.assertEqual(original_id, consumer['id'])

    def test_consumer_update_bad_secret(self):
        consumer = self._create_single_consumer()
        original_id = consumer['id']
        update_ref = copy.deepcopy(consumer)
        update_ref['description'] = uuid.uuid4().hex
        update_ref['secret'] = uuid.uuid4().hex
        self.patch(self.CONSUMER_URL + '/%s' % original_id, body={'consumer': update_ref}, expected_status=http.client.BAD_REQUEST)

    def test_consumer_update_bad_id(self):
        consumer = self._create_single_consumer()
        original_id = consumer['id']
        original_description = consumer['description']
        update_description = original_description + '_new'
        update_ref = copy.deepcopy(consumer)
        update_ref['description'] = update_description
        update_ref['id'] = update_description
        self.patch(self.CONSUMER_URL + '/%s' % original_id, body={'consumer': update_ref}, expected_status=http.client.BAD_REQUEST)

    def test_consumer_update_normalize_field(self):
        field1_name = 'some:weird-field'
        field1_orig_value = uuid.uuid4().hex
        extra_fields = {field1_name: field1_orig_value}
        consumer = self._consumer_create(**extra_fields)
        consumer_id = consumer['id']
        field1_new_value = uuid.uuid4().hex
        field2_name = 'weird:some-field'
        field2_value = uuid.uuid4().hex
        update_ref = {field1_name: field1_new_value, field2_name: field2_value}
        update_resp = self.patch(self.CONSUMER_URL + '/%s' % consumer_id, body={'consumer': update_ref})
        consumer = update_resp.result['consumer']
        normalized_field1_name = 'some_weird_field'
        self.assertEqual(field1_new_value, consumer[normalized_field1_name])
        normalized_field2_name = 'weird_some_field'
        self.assertEqual(field2_value, consumer[normalized_field2_name])

    def test_consumer_create_no_description(self):
        resp = self.post(self.CONSUMER_URL, body={'consumer': {}})
        consumer = resp.result['consumer']
        consumer_id = consumer['id']
        self.assertIsNone(consumer['description'])
        self.assertIsNotNone(consumer_id)
        self.assertIsNotNone(consumer['secret'])

    def test_consumer_get_bad_id(self):
        url = self.CONSUMER_URL + '/%(consumer_id)s' % {'consumer_id': uuid.uuid4().hex}
        self.get(url, expected_status=http.client.NOT_FOUND)
        self.head(url, expected_status=http.client.NOT_FOUND)