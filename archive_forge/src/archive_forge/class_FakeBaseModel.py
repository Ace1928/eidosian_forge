import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeBaseModel(object):

    def __repr__(self):
        return '<' + self.__class__.model_name + '%s>' % self._info