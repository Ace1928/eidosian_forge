import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class FakeEnumAlt(fields.Enum):
    FROG = 'frog'
    PLATYPUS = 'platypus'
    AARDVARK = 'aardvark'
    ALL = set([FROG, PLATYPUS, AARDVARK])

    def __init__(self, **kwargs):
        super(FakeEnumAlt, self).__init__(valid_values=FakeEnumAlt.ALL, **kwargs)