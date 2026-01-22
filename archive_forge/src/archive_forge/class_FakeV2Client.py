from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import notificationtypes
from monascaclient.v2_0 import shell
class FakeV2Client(object):

    def __init__(self):
        super(FakeV2Client, self).__init__()
        self.notificationtypes = mock.Mock(spec=notificationtypes.NotificationTypesManager)