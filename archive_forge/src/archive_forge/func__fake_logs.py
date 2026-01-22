import io
import logging
import os
import pprint
import sys
import typing as ty
import fixtures
from oslotest import base
import testtools.content
from openstack.tests import fixtures as os_fixtures
from openstack import utils
def _fake_logs(self):
    pass