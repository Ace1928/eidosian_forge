import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class FakeGateway(object):

    def __init__(self, db=None, store=None, notifier=None, policy=None, repo=None):
        self.db = db
        self.store = store
        self.notifier = notifier
        self.policy = policy
        self.repo = repo

    def get_repo(self, context):
        return self.repo