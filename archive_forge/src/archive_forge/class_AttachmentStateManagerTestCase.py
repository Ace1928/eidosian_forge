from unittest import mock
from oslo_config import cfg
from oslotest import base
from cinderclient import exceptions as cinder_exception
from glance_store.common import attachment_state_manager as attach_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
class AttachmentStateManagerTestCase(base.BaseTestCase):

    class FakeAttachmentState:

        def __init__(self):
            self.attachments = {mock.sentinel.attachments}

    def setUp(self):
        super(AttachmentStateManagerTestCase, self).setUp()
        self.__manager__ = attach_manager.__manager__

    def get_state(self):
        with self.__manager__.get_state() as state:
            return state

    def test_get_state_host_not_initialized(self):
        self.__manager__.state = None
        self.assertRaises(exceptions.HostNotInitialized, self.get_state)

    def test_get_state(self):
        self.__manager__.state = self.FakeAttachmentState()
        state = self.get_state()
        self.assertEqual({mock.sentinel.attachments}, state.attachments)