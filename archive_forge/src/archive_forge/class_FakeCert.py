import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeCert(object):

    def __init__(self, pem):
        self.pem = pem