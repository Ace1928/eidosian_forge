import functools
from oslotest import base as test_base
from oslo_utils import reflection
class MemberGetTest(test_base.BaseTestCase):

    def test_get_members_exclude_hidden(self):
        obj = TestObject()
        members = list(reflection.get_members(obj, exclude_hidden=True))
        self.assertEqual(1, len(members))

    def test_get_members_no_exclude_hidden(self):
        obj = TestObject()
        members = list(reflection.get_members(obj, exclude_hidden=False))
        self.assertGreater(len(members), 1)

    def test_get_members_names_exclude_hidden(self):
        obj = TestObject()
        members = list(reflection.get_member_names(obj, exclude_hidden=True))
        self.assertEqual(['hi'], members)

    def test_get_members_names_no_exclude_hidden(self):
        obj = TestObject()
        members = list(reflection.get_member_names(obj, exclude_hidden=False))
        members = [member for member in members if not member.startswith('__')]
        self.assertEqual(['_hello', 'hi'], sorted(members))