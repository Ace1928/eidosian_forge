from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class FakeImageMembershipFactory(object):

    def __init__(self, result=None):
        self.result = None
        self.image = None
        self.member_id = None

    def new_image_member(self, image, member_id):
        self.image = image
        self.member_id = member_id
        return self.result