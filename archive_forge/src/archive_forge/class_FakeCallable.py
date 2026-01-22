import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
class FakeCallable(object):

    def __call__(self, *args, **kwargs):
        return 'pass'