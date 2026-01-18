import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
@mock.patch('oslo_utils.eventletutils._patcher')
def test_partially_patched_warning(self, mock_patcher):
    is_patched = set()
    mock_patcher.already_patched = True
    mock_patcher.is_monkey_patched.side_effect = lambda m: m in is_patched
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        eventletutils.warn_eventlet_not_patched(['os'])
    self.assertEqual(1, len(capture))
    is_patched.add('os')
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        eventletutils.warn_eventlet_not_patched(['os'])
    self.assertEqual(0, len(capture))
    is_patched.add('thread')
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        eventletutils.warn_eventlet_not_patched(['os', 'thread'])
    self.assertEqual(0, len(capture))
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        eventletutils.warn_eventlet_not_patched(['all'])
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(RuntimeWarning, w.category)
    for m in ['os', 'thread']:
        self.assertNotIn(m, str(w.message))