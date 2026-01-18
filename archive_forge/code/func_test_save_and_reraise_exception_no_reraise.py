import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_save_and_reraise_exception_no_reraise(self):
    """Test that suppressing the reraise works."""
    try:
        raise Exception('foo')
    except Exception:
        with excutils.save_and_reraise_exception() as ctxt:
            ctxt.reraise = False