import io
import os
import re
import struct
import subprocess
import tempfile
from unittest import mock
from oslo_utils import units
from glance.common import format_inspector
from glance.tests import utils as test_utils
def test_vhdx_invalid(self):
    self._test_format_with_invalid_data('vhdx')