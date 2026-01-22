from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
@registry.has_registry_receivers
class AnotherObjectWithDecoratedCallback(ObjectWithDecoratedCallback, MixinWithNew):

    def __init__(self):
        super(AnotherObjectWithDecoratedCallback, self).__init__()
        self.counter2 = 0

    @registry.receives(resources.NETWORK, [events.AFTER_DELETE], PRI_CALLBACK)
    def callback2(self, *args, **kwargs):
        self.counter2 += 1