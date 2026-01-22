import asyncio
import concurrent.futures
import copy
import datetime
import functools
import os
import re
import sys
import threading
import warnings
from base64 import b64decode, b64encode
from queue import Empty
from typing import Any
from unittest.mock import MagicMock, Mock
import nbformat
import pytest
import xmltodict
from flaky import flaky  # type:ignore
from jupyter_client import KernelClient, KernelManager
from jupyter_client._version import version_info
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert.filters import strip_ansi
from nbformat import NotebookNode
from testpath import modified_env
from traitlets import TraitError
from nbclient import NotebookClient, execute
from nbclient.exceptions import CellExecutionError
from .base import NBClientTestsBase
class TestRunCell(NBClientTestsBase):
    """Contains test functions for NotebookClient.execute_cell"""

    @prepare_cell_mocks()
    def test_idle_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 1
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'execute_reply'}, 'parent_header': {'msg_id': 'wrong_parent'}, 'content': {'name': 'stdout', 'text': 'foo'}})
    def test_message_for_wrong_parent(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'status', 'header': {'msg_type': 'status'}, 'content': {'execution_state': 'busy'}})
    def test_busy_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stderr', 'text': 'bar'}})
    def test_deadline_exec_reply(self, executor, cell_mock, message_mock):

        async def get_msg(timeout):
            await asyncio.sleep(timeout)
            raise Empty
        executor.kc.shell_channel.get_msg = get_msg
        executor.timeout = 1
        with pytest.raises(TimeoutError):
            executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 3
        self.assertListEqual(cell_mock.outputs, [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo'}, {'output_type': 'stream', 'name': 'stderr', 'text': 'bar'}])

    @prepare_cell_mocks()
    def test_deadline_iopub(self, executor, cell_mock, message_mock):
        message_mock.side_effect = Empty()
        executor.raise_on_iopub_timeout = True
        with pytest.raises(TimeoutError):
            executor.execute_cell(cell_mock, 0)

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stderr', 'text': 'bar'}})
    def test_eventual_deadline_iopub(self, executor, cell_mock, message_mock):

        def message_seq(messages):
            yield from messages
            while True:
                yield Empty()
        message_mock.side_effect = message_seq(list(message_mock.side_effect)[:-1])
        executor.kc.shell_channel.get_msg = Mock(return_value=make_future({'parent_header': {'msg_id': executor.parent_id}}))
        executor.raise_on_iopub_timeout = True
        with pytest.raises(TimeoutError):
            executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count >= 3
        self.assertListEqual(cell_mock.outputs, [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo'}, {'output_type': 'stream', 'name': 'stderr', 'text': 'bar'}])

    @prepare_cell_mocks({'msg_type': 'execute_input', 'header': {'msg_type': 'execute_input'}, 'content': {}})
    def test_execute_input_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stderr', 'text': 'bar'}})
    def test_stream_messages(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 3
        self.assertListEqual(cell_mock.outputs, [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo'}, {'output_type': 'stream', 'name': 'stderr', 'text': 'bar'}])

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'execute_reply'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'clear_output', 'header': {'msg_type': 'clear_output'}, 'content': {}})
    def test_clear_output_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 3
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'clear_output', 'header': {'msg_type': 'clear_output'}, 'content': {'wait': True}})
    def test_clear_output_wait_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 3
        self.assertTrue(executor.clear_before_next_output)
        assert cell_mock.outputs == [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo'}]

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'clear_output', 'header': {'msg_type': 'clear_output'}, 'content': {'wait': True}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stderr', 'text': 'bar'}})
    def test_clear_output_wait_then_message_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 4
        assert not executor.clear_before_next_output
        assert cell_mock.outputs == [{'output_type': 'stream', 'name': 'stderr', 'text': 'bar'}]

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo'}}, {'msg_type': 'clear_output', 'header': {'msg_type': 'clear_output'}, 'content': {'wait': True}}, {'msg_type': 'update_display_data', 'header': {'msg_type': 'update_display_data'}, 'content': {'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}})
    def test_clear_output_wait_then_update_display_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 4
        assert executor.clear_before_next_output
        assert cell_mock.outputs == [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo'}]

    @prepare_cell_mocks({'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'execution_count': 42}})
    def test_execution_count_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.execution_count == 42
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'execution_count': 42}})
    def test_execution_count_message_ignored_on_override(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0, execution_count=21)
        assert message_mock.call_count == 2
        assert cell_mock.execution_count == 21
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'execution_count': 42, 'name': 'stdout', 'text': 'foo'}})
    def test_execution_count_with_stream_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.execution_count == 42
        assert cell_mock.outputs == [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo'}]

    @prepare_cell_mocks({'msg_type': 'comm', 'header': {'msg_type': 'comm'}, 'content': {'comm_id': 'foobar', 'data': {'state': {'foo': 'bar'}}}})
    def test_widget_comm_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        self.assertEqual(executor.widget_state, {'foobar': {'foo': 'bar'}})
        assert not executor.widget_buffers
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'comm', 'header': {'msg_type': 'comm'}, 'buffers': [b'123'], 'content': {'comm_id': 'foobar', 'data': {'state': {'foo': 'bar'}, 'buffer_paths': [['path']]}}})
    def test_widget_comm_buffer_message_single(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert executor.widget_state == {'foobar': {'foo': 'bar'}}
        assert executor.widget_buffers == {'foobar': {('path',): {'data': 'MTIz', 'encoding': 'base64', 'path': ['path']}}}
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'comm', 'header': {'msg_type': 'comm'}, 'buffers': [b'123'], 'content': {'comm_id': 'foobar', 'data': {'state': {'foo': 'bar'}, 'buffer_paths': [['path']]}}}, {'msg_type': 'comm', 'header': {'msg_type': 'comm'}, 'buffers': [b'123'], 'content': {'comm_id': 'foobar', 'data': {'state': {'foo2': 'bar2'}, 'buffer_paths': [['path2']]}}})
    def test_widget_comm_buffer_messages(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 3
        assert executor.widget_state == {'foobar': {'foo': 'bar', 'foo2': 'bar2'}}
        assert executor.widget_buffers == {'foobar': {('path',): {'data': 'MTIz', 'encoding': 'base64', 'path': ['path']}, ('path2',): {'data': 'MTIz', 'encoding': 'base64', 'path': ['path2']}}}
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'comm', 'header': {'msg_type': 'comm'}, 'content': {'comm_id': 'foobar', 'data': {'foo': 'bar'}}})
    def test_unknown_comm_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert not executor.widget_state
        assert not executor.widget_buffers
        assert cell_mock.outputs == []

    @prepare_cell_mocks({'msg_type': 'execute_result', 'header': {'msg_type': 'execute_result'}, 'content': {'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}, 'execution_count': 42}})
    def test_execute_result_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.execution_count == 42
        assert cell_mock.outputs == [{'output_type': 'execute_result', 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}, 'execution_count': 42}]
        assert not executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'execute_result', 'header': {'msg_type': 'execute_result'}, 'content': {'transient': {'display_id': 'foobar'}, 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}, 'execution_count': 42}})
    def test_execute_result_with_display_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.execution_count == 42
        assert cell_mock.outputs == [{'output_type': 'execute_result', 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}, 'execution_count': 42}]
        assert 'foobar' in executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'display_data', 'header': {'msg_type': 'display_data'}, 'content': {'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}})
    def test_display_data_without_id_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == [{'output_type': 'display_data', 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}]
        assert not executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'display_data', 'header': {'msg_type': 'display_data'}, 'content': {'transient': {'display_id': 'foobar'}, 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}})
    def test_display_data_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == [{'output_type': 'display_data', 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}]
        assert 'foobar' in executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'display_data', 'header': {'msg_type': 'display_data'}, 'content': {'transient': {'display_id': 'foobar'}, 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}}, {'msg_type': 'display_data', 'header': {'msg_type': 'display_data'}, 'content': {'transient': {'display_id': 'foobar_other'}, 'metadata': {'metafoo_other': 'metabar_other'}, 'data': {'foo': 'bar_other'}}}, {'msg_type': 'display_data', 'header': {'msg_type': 'display_data'}, 'content': {'transient': {'display_id': 'foobar'}, 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}})
    def test_display_data_same_id_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 4
        assert cell_mock.outputs == [{'output_type': 'display_data', 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}, {'output_type': 'display_data', 'metadata': {'metafoo_other': 'metabar_other'}, 'data': {'foo': 'bar_other'}}, {'output_type': 'display_data', 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}]
        assert 'foobar' in executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'update_display_data', 'header': {'msg_type': 'update_display_data'}, 'content': {'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}})
    def test_update_display_data_without_id_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == []
        assert not executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'display_data', 'header': {'msg_type': 'display_data'}, 'content': {'transient': {'display_id': 'foobar'}, 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}}, {'msg_type': 'update_display_data', 'header': {'msg_type': 'update_display_data'}, 'content': {'transient': {'display_id': 'foobar2'}, 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}})
    def test_update_display_data_mismatch_id_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 3
        assert cell_mock.outputs == [{'output_type': 'display_data', 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}]
        assert 'foobar' in executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'display_data', 'header': {'msg_type': 'display_data'}, 'content': {'transient': {'display_id': 'foobar'}, 'metadata': {'metafoo': 'metabar'}, 'data': {'foo': 'bar'}}}, {'msg_type': 'update_display_data', 'header': {'msg_type': 'update_display_data'}, 'content': {'transient': {'display_id': 'foobar'}, 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}})
    def test_update_display_data_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 3
        assert cell_mock.outputs == [{'output_type': 'display_data', 'metadata': {'metafoo2': 'metabar2'}, 'data': {'foo': 'bar2', 'baz': 'foobarbaz'}}]
        assert 'foobar' in executor._display_id_map

    @prepare_cell_mocks({'msg_type': 'error', 'header': {'msg_type': 'error'}, 'content': {'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}})
    def test_error_message(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == [{'output_type': 'error', 'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}]

    @prepare_cell_mocks({'msg_type': 'error', 'header': {'msg_type': 'error'}, 'content': {'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}}, reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_error_and_error_status_messages(self, executor, cell_mock, message_mock):
        with self.assertRaises(CellExecutionError):
            executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == [{'output_type': 'error', 'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}]

    @prepare_cell_mocks({'msg_type': 'error', 'header': {'msg_type': 'error'}, 'content': {'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}}, reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'ok'}})
    def test_error_message_only(self, executor, cell_mock, message_mock):
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 2
        assert cell_mock.outputs == [{'output_type': 'error', 'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}]

    @prepare_cell_mocks(reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_allow_errors(self, executor, cell_mock, message_mock):
        executor.allow_errors = True
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 1
        assert cell_mock.outputs == []

    @prepare_cell_mocks(reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error', 'ename': 'NotImplementedError'}})
    def test_allow_error_names(self, executor, cell_mock, message_mock):
        executor.allow_error_names = ['NotImplementedError']
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 1
        assert cell_mock.outputs == []

    @prepare_cell_mocks(reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_raises_exception_tag(self, executor, cell_mock, message_mock):
        cell_mock.metadata['tags'] = ['raises-exception']
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 1
        assert cell_mock.outputs == []

    @prepare_cell_mocks(reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_non_code_cell(self, executor, cell_mock, message_mock):
        cell_mock = NotebookNode(source='"foo" = "bar"', metadata={}, cell_type='raw', outputs=[])
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 0
        assert cell_mock.outputs == []

    @prepare_cell_mocks(reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_no_source(self, executor, cell_mock, message_mock):
        cell_mock = NotebookNode(source='     ', metadata={}, cell_type='code', outputs=[])
        executor.execute_cell(cell_mock, 0)
        assert message_mock.call_count == 0
        assert cell_mock.outputs == []

    @prepare_cell_mocks()
    def test_cell_hooks(self, executor, cell_mock, message_mock):
        executor, hooks = get_executor_with_hooks(executor=executor)
        executor.execute_cell(cell_mock, 0)
        hooks['on_cell_start'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_execute'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_complete'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_executed'].assert_called_once_with(cell=cell_mock, cell_index=0, execute_reply=EXECUTE_REPLY_OK)
        hooks['on_cell_error'].assert_not_called()
        hooks['on_notebook_start'].assert_not_called()
        hooks['on_notebook_complete'].assert_not_called()
        hooks['on_notebook_error'].assert_not_called()

    @prepare_cell_mocks({'msg_type': 'error', 'header': {'msg_type': 'error'}, 'content': {'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}}, reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_error_cell_hooks(self, executor, cell_mock, message_mock):
        executor, hooks = get_executor_with_hooks(executor=executor)
        with self.assertRaises(CellExecutionError):
            executor.execute_cell(cell_mock, 0)
        hooks['on_cell_start'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_execute'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_complete'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_executed'].assert_called_once_with(cell=cell_mock, cell_index=0, execute_reply=EXECUTE_REPLY_ERROR)
        hooks['on_cell_error'].assert_called_once_with(cell=cell_mock, cell_index=0, execute_reply=EXECUTE_REPLY_ERROR)
        hooks['on_notebook_start'].assert_not_called()
        hooks['on_notebook_complete'].assert_not_called()
        hooks['on_notebook_error'].assert_not_called()

    @prepare_cell_mocks(reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_non_code_cell_hooks(self, executor, cell_mock, message_mock):
        cell_mock = NotebookNode(source='"foo" = "bar"', metadata={}, cell_type='raw', outputs=[])
        executor, hooks = get_executor_with_hooks(executor=executor)
        executor.execute_cell(cell_mock, 0)
        hooks['on_cell_start'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_execute'].assert_not_called()
        hooks['on_cell_complete'].assert_not_called()
        hooks['on_cell_executed'].assert_not_called()
        hooks['on_cell_error'].assert_not_called()
        hooks['on_notebook_start'].assert_not_called()
        hooks['on_notebook_complete'].assert_not_called()
        hooks['on_notebook_error'].assert_not_called()

    @prepare_cell_mocks()
    def test_async_cell_hooks(self, executor, cell_mock, message_mock):
        executor, hooks = get_executor_with_hooks(executor=executor, async_hooks=True)
        executor.execute_cell(cell_mock, 0)
        hooks['on_cell_start'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_execute'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_complete'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_executed'].assert_called_once_with(cell=cell_mock, cell_index=0, execute_reply=EXECUTE_REPLY_OK)
        hooks['on_cell_error'].assert_not_called()
        hooks['on_notebook_start'].assert_not_called()
        hooks['on_notebook_complete'].assert_not_called()
        hooks['on_notebook_error'].assert_not_called()

    @prepare_cell_mocks({'msg_type': 'error', 'header': {'msg_type': 'error'}, 'content': {'ename': 'foo', 'evalue': 'bar', 'traceback': ['Boom']}}, reply_msg={'msg_type': 'execute_reply', 'header': {'msg_type': 'execute_reply'}, 'content': {'status': 'error'}})
    def test_error_async_cell_hooks(self, executor, cell_mock, message_mock):
        executor, hooks = get_executor_with_hooks(executor=executor, async_hooks=True)
        with self.assertRaises(CellExecutionError):
            executor.execute_cell(cell_mock, 0)
        hooks['on_cell_start'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_execute'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_complete'].assert_called_once_with(cell=cell_mock, cell_index=0)
        hooks['on_cell_executed'].assert_called_once_with(cell=cell_mock, cell_index=0, execute_reply=EXECUTE_REPLY_ERROR)
        hooks['on_cell_error'].assert_called_once_with(cell=cell_mock, cell_index=0, execute_reply=EXECUTE_REPLY_ERROR)
        hooks['on_notebook_start'].assert_not_called()
        hooks['on_notebook_complete'].assert_not_called()
        hooks['on_notebook_error'].assert_not_called()

    @prepare_cell_mocks({'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo1'}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stderr', 'text': 'bar1'}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stdout', 'text': 'foo2'}}, {'msg_type': 'stream', 'header': {'msg_type': 'stream'}, 'content': {'name': 'stderr', 'text': 'bar2'}})
    def test_coalesce_streams(self, executor, cell_mock, message_mock):
        executor.coalesce_streams = True
        executor.execute_cell(cell_mock, 0)
        assert cell_mock.outputs == [{'output_type': 'stream', 'name': 'stdout', 'text': 'foo1foo2'}, {'output_type': 'stream', 'name': 'stderr', 'text': 'bar1bar2'}]