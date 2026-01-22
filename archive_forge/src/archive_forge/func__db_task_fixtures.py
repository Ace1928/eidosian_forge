import datetime
import hashlib
import http.client as http
import os
import requests
from unittest import mock
import uuid
from castellan.common import exception as castellan_exception
import glance_store as store
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import fixture
import testtools
import webob
import webob.exc
import glance.api.v2.image_actions
import glance.api.v2.images
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance import domain
import glance.notifier
import glance.schema
from glance.tests.unit import base
from glance.tests.unit.keymgr import fake as fake_keymgr
import glance.tests.unit.utils as unit_test_utils
from glance.tests.unit.v2 import test_tasks_resource
import glance.tests.utils as test_utils
def _db_task_fixtures(task_id, **kwargs):
    default_datetime = timeutils.utcnow()
    obj = {'id': task_id, 'status': kwargs.get('status', 'pending'), 'type': 'import', 'input': kwargs.get('input', {}), 'result': None, 'owner': None, 'image_id': kwargs.get('image_id'), 'user_id': kwargs.get('user_id'), 'request_id': kwargs.get('request_id'), 'message': None, 'expires_at': default_datetime + datetime.timedelta(days=365), 'created_at': default_datetime, 'updated_at': default_datetime, 'deleted_at': None, 'deleted': False}
    obj.update(kwargs)
    return obj