import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_replication_compare_with_bad_args(self):
    args = ['aaa', 'bbb']
    command = glance_replicator.replication_compare
    self.assertTrue(check_bad_args(command, args))