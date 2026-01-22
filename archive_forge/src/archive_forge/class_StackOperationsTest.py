from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
class StackOperationsTest(testtools.TestCase):

    def test_delete_stack(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.delete()
        manager.delete.assert_called_once_with('the_stack/abcd1234')

    def test_abandon_stack(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.abandon()
        manager.abandon.assert_called_once_with('the_stack/abcd1234')

    def test_get_stack(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.get()
        manager.get.assert_called_once_with('the_stack/abcd1234')

    def test_update_stack(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.update()
        manager.update.assert_called_once_with('the_stack/abcd1234')

    def test_create_stack(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack = stack.create()
        manager.create.assert_called_once_with('the_stack/abcd1234')

    def test_preview_stack(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack = stack.preview()
        manager.preview.assert_called_once_with()

    def test_snapshot(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.snapshot('foo')
        manager.snapshot.assert_called_once_with('the_stack/abcd1234', 'foo')

    def test_snapshot_show(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.snapshot_show('snap1234')
        manager.snapshot_show.assert_called_once_with('the_stack/abcd1234', 'snap1234')

    def test_snapshot_delete(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.snapshot_delete('snap1234')
        manager.snapshot_delete.assert_called_once_with('the_stack/abcd1234', 'snap1234')

    def test_restore(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.restore('snap1234')
        manager.restore.assert_called_once_with('the_stack/abcd1234', 'snap1234')

    def test_snapshot_list(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.snapshot_list()
        manager.snapshot_list.assert_called_once_with('the_stack/abcd1234')

    def test_output_list(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.output_list()
        manager.output_list.assert_called_once_with('the_stack/abcd1234')

    def test_output_show(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'the_stack', 'abcd1234')
        stack.output_show('out123')
        manager.output_show.assert_called_once_with('the_stack/abcd1234', 'out123')

    def test_environment_show(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'env_stack', 'env1')
        stack.environment()
        manager.environment.assert_called_once_with('env_stack/env1')

    def test_files_show(self):
        manager = mock.MagicMock()
        stack = mock_stack(manager, 'files_stack', 'files1')
        stack.files()
        manager.files.assert_called_once_with('files_stack/files1')