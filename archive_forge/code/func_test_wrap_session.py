import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@mock.patch('osprofiler.sqlalchemy.handle_error')
@mock.patch('osprofiler.sqlalchemy._before_cursor_execute')
@mock.patch('osprofiler.sqlalchemy._after_cursor_execute')
def test_wrap_session(self, mock_after_exc, mock_before_exc, mock_handle_error):
    sa = mock.MagicMock()

    @contextlib.contextmanager
    def _session():
        session = mock.MagicMock()
        session.bind = mock.MagicMock()
        session.bind.traced = None
        yield session
    mock_before_exc.return_value = 'before'
    mock_after_exc.return_value = 'after'
    session = sqlalchemy.wrap_session(sa, _session())
    with session as sess:
        pass
    mock_before_exc.assert_called_once_with('db')
    mock_after_exc.assert_called_once_with(hide_result=True)
    expected_calls = [mock.call(sess.bind, 'before_cursor_execute', 'before'), mock.call(sess.bind, 'after_cursor_execute', 'after'), mock.call(sess.bind, 'handle_error', mock_handle_error)]
    self.assertEqual(sa.event.listen.call_args_list, expected_calls)