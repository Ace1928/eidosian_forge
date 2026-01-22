import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
class TestImageMemberNotifications(utils.BaseTestCase):
    """Test Image Member Notifications work"""

    def setUp(self):
        super(TestImageMemberNotifications, self).setUp()
        self.context = glance.context.RequestContext(tenant=TENANT2, user=USER1)
        self.notifier = unit_test_utils.FakeNotifier()
        self.image = ImageStub(image_id=UUID1, name='image-1', status='active', size=1024, created_at=DATETIME, updated_at=DATETIME, owner=TENANT1, visibility='public', container_format='ami', tags=['one', 'two'], disk_format='ami', min_ram=128, min_disk=10, checksum='ca425b88f047ce8ec45ee90e813ada91', locations=['http://127.0.0.1'])
        self.image_member = glance.domain.ImageMembership(id=1, image_id=UUID1, member_id=TENANT1, created_at=DATETIME, updated_at=DATETIME, status='accepted')
        self.image_member_repo_stub = ImageMemberRepoStub()
        self.image_member_repo_proxy = glance.notifier.ImageMemberRepoProxy(self.image_member_repo_stub, self.image, self.context, self.notifier)
        self.image_member_proxy = glance.notifier.ImageMemberProxy(self.image_member, self.context, self.notifier)

    def _assert_image_member_with_notifier(self, output_log, deleted=False):
        self.assertEqual(self.image_member.member_id, output_log['payload']['member_id'])
        self.assertEqual(self.image_member.image_id, output_log['payload']['image_id'])
        self.assertEqual(self.image_member.status, output_log['payload']['status'])
        self.assertEqual(timeutils.isotime(self.image_member.created_at), output_log['payload']['created_at'])
        self.assertEqual(timeutils.isotime(self.image_member.updated_at), output_log['payload']['updated_at'])
        if deleted:
            self.assertTrue(output_log['payload']['deleted'])
            self.assertIsNotNone(output_log['payload']['deleted_at'])
        else:
            self.assertFalse(output_log['payload']['deleted'])
            self.assertIsNone(output_log['payload']['deleted_at'])

    def test_image_member_add_notification(self):
        self.image_member_repo_proxy.add(self.image_member_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.member.create', output_log['event_type'])
        self._assert_image_member_with_notifier(output_log)

    def test_image_member_add_notification_disabled(self):
        self.config(disabled_notifications=['image.member.create'])
        self.image_member_repo_proxy.add(self.image_member_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_member_save_notification(self):
        self.image_member_repo_proxy.save(self.image_member_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.member.update', output_log['event_type'])
        self._assert_image_member_with_notifier(output_log)

    def test_image_member_save_notification_disabled(self):
        self.config(disabled_notifications=['image.member.update'])
        self.image_member_repo_proxy.save(self.image_member_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_member_delete_notification(self):
        self.image_member_repo_proxy.remove(self.image_member_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.member.delete', output_log['event_type'])
        self._assert_image_member_with_notifier(output_log, deleted=True)

    def test_image_member_delete_notification_disabled(self):
        self.config(disabled_notifications=['image.member.delete'])
        self.image_member_repo_proxy.remove(self.image_member_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_member_get(self):
        image_member = self.image_member_repo_proxy.get(TENANT1)
        self.assertIsInstance(image_member, glance.notifier.ImageMemberProxy)
        self.assertEqual('image_member_from_get', image_member.repo)

    def test_image_member_list(self):
        image_members = self.image_member_repo_proxy.list()
        self.assertIsInstance(image_members[0], glance.notifier.ImageMemberProxy)
        self.assertEqual('image_members_from_list', image_members[0].repo)