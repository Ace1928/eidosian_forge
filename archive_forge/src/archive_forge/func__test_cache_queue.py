import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
def _test_cache_queue(self, supported=True, forbidden=False):
    args = argparse.Namespace(id=['image1'])
    with mock.patch.object(self.gc.cache, 'queue') as mocked_cache_queue:
        if supported:
            mocked_cache_queue.return_value = None
        else:
            mocked_cache_queue.side_effect = exc.HTTPNotImplemented
        if forbidden:
            mocked_cache_queue.side_effect = exc.HTTPForbidden
        test_shell.do_cache_queue(self.gc, args)
        if supported:
            mocked_cache_queue.assert_called_once_with('image1')