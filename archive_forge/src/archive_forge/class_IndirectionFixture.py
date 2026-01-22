from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
class IndirectionFixture(fixtures.Fixture):

    def __init__(self, indirection_api=None):
        self.indirection_api = indirection_api or FakeIndirectionAPI()

    def setUp(self):
        super(IndirectionFixture, self).setUp()
        self.useFixture(fixtures.MonkeyPatch('oslo_versionedobjects.base.VersionedObject.indirection_api', self.indirection_api))