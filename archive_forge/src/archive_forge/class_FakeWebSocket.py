from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
class FakeWebSocket(object):

    def __init__(self, events):
        self.events = events

    def recv(self):
        return self.events.pop(0)