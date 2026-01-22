import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class FakeCertificatesModelManager(FakeBaseModelManager):

    def get(self, cluster_uuid):
        pass