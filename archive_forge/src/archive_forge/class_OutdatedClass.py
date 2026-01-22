from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@versionutils.deprecated(as_of=versionutils.deprecated.OCATA, remove_in=+2)
class OutdatedClass(object):
    pass