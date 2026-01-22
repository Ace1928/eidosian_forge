from unittest import mock
from keystoneauth1 import plugin
class FakeEventSink(object):

    def __init__(self, evt):
        self.events = []
        self.evt = evt

    def consume(self, stack, event):
        self.events.append(event)
        self.evt.send(None)