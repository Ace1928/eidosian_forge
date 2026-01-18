import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_unified_not_added(self):
    ret_opts = diff.default_style_unified(diff_opts=['{}'.format(self.style)])
    self.assertEqual(['{}'.format(self.style)], ret_opts)