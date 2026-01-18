import copy
import io
from unittest import mock
from osc_lib import exceptions
from requests_mock.contrib import fixture
from openstackclient.object.v1 import object as object_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_save_to_stdout(self):
    self.requests_mock.register_uri('GET', object_fakes.ENDPOINT + '/' + object_fakes.container_name + '/' + object_fakes.object_name_1, status_code=200, content=object_fakes.object_1_content)
    arglist = [object_fakes.container_name, object_fakes.object_name_1, '--file', '-']
    verifylist = [('container', object_fakes.container_name), ('object', object_fakes.object_name_1), ('file', '-')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)

    class FakeStdout(io.BytesIO):

        def __init__(self):
            io.BytesIO.__init__(self)
            self.context_manager_calls = []

        def __enter__(self):
            self.context_manager_calls.append('__enter__')
            return self

        def __exit__(self, *a):
            self.context_manager_calls.append('__exit__')
    with mock.patch('sys.stdout') as fake_stdout, mock.patch('os.fdopen', return_value=FakeStdout()) as fake_fdopen:
        fake_stdout.fileno.return_value = 123
        self.cmd.take_action(parsed_args)
    self.assertEqual(fake_fdopen.return_value.getvalue(), object_fakes.object_1_content)
    self.assertEqual(fake_fdopen.mock_calls, [mock.call(123, 'wb')])
    self.assertEqual(fake_fdopen.return_value.context_manager_calls, ['__enter__', '__exit__'])