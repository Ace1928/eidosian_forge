import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def test_set_msg_id_when_capturing(self):
    msg_id = 'msg-id'
    get_ipython = self._mock_get_ipython(msg_id)
    clear_output = self._mock_clear_output()
    with self._mocked_ipython(get_ipython, clear_output):
        widget = widget_output.Output()
        assert widget.msg_id == ''
        with widget:
            assert widget.msg_id == msg_id
        assert widget.msg_id == ''