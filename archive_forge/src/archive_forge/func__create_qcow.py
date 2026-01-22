import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
def _create_qcow(self, size):
    fn = tempfile.mktemp(prefix='glance-unittest-images-', suffix='.qcow2')
    subprocess.check_output('qemu-img create -f qcow2 %s %i' % (fn, size), shell=True)
    return fn