import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
class TestInitAndCleanup(base.TestBase):

    def test_initialize_app(self):
        app, command = make_app()
        app.initialize_app = mock.MagicMock(name='initialize_app')
        app.run(['mock'])
        app.initialize_app.assert_called_once_with(['mock'])

    def test_prepare_to_run_command(self):
        app, command = make_app()
        app.prepare_to_run_command = mock.MagicMock(name='prepare_to_run_command')
        app.run(['mock'])
        app.prepare_to_run_command.assert_called_once_with(command())

    def test_interrupt_command(self):
        app, command = make_app()
        result = app.run(['interrupt'])
        self.assertEqual(result, 130)

    def test_pipeclose_command(self):
        app, command = make_app()
        result = app.run(['pipe-close'])
        self.assertEqual(result, 141)

    def test_clean_up_success(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up')
        ret = app.run(['mock'])
        app.clean_up.assert_called_once_with(command.return_value, 0, None)
        self.assertEqual(ret, 0)

    def test_clean_up_error(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up')
        ret = app.run(['error'])
        self.assertNotEqual(ret, 0)
        app.clean_up.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 1, mock.ANY), call_args)
        args, kwargs = call_args
        self.assertIsInstance(args[2], RuntimeError)
        self.assertEqual(('test exception',), args[2].args)

    def test_clean_up_error_debug(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up')
        ret = app.run(['--debug', 'error'])
        self.assertNotEqual(ret, 0)
        self.assertTrue(app.clean_up.called)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 1, mock.ANY), call_args)
        args, kwargs = call_args
        self.assertIsInstance(args[2], RuntimeError)
        self.assertEqual(('test exception',), args[2].args)

    def test_clean_up_interrupt(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up')
        ret = app.run(['interrupt'])
        self.assertNotEqual(ret, 0)
        app.clean_up.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 130, mock.ANY), call_args)
        args, kwargs = call_args
        self.assertIsInstance(args[2], KeyboardInterrupt)

    def test_clean_up_pipeclose(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up')
        ret = app.run(['pipe-close'])
        self.assertNotEqual(ret, 0)
        app.clean_up.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 141, mock.ANY), call_args)
        args, kwargs = call_args
        self.assertIsInstance(args[2], BrokenPipeError)

    def test_error_handling_clean_up_raises_exception(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up', side_effect=RuntimeError('within clean_up'))
        app.run(['error'])
        self.assertTrue(app.clean_up.called)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 1, mock.ANY), call_args)
        args, kwargs = call_args
        self.assertIsInstance(args[2], RuntimeError)
        self.assertEqual(('test exception',), args[2].args)

    def test_error_handling_clean_up_raises_exception_debug(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up', side_effect=RuntimeError('within clean_up'))
        try:
            ret = app.run(['--debug', 'error'])
        except RuntimeError as err:
            if not hasattr(err, '__context__'):
                self.assertIsNot(err, app.clean_up.call_args_list[0][0][2])
        else:
            self.assertNotEqual(ret, 0)
        self.assertTrue(app.clean_up.called)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 1, mock.ANY), call_args)
        args, kwargs = call_args
        self.assertIsInstance(args[2], RuntimeError)
        self.assertEqual(('test exception',), args[2].args)

    def test_normal_clean_up_raises_exception(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up', side_effect=RuntimeError('within clean_up'))
        app.run(['mock'])
        self.assertTrue(app.clean_up.called)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 0, None), call_args)

    def test_normal_clean_up_raises_exception_debug(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up', side_effect=RuntimeError('within clean_up'))
        app.run(['--debug', 'mock'])
        self.assertTrue(app.clean_up.called)
        call_args = app.clean_up.call_args_list[0]
        self.assertEqual(mock.call(mock.ANY, 0, None), call_args)