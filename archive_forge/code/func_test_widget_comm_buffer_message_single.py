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
@prepare_cell_mocks({'msg_type': 'comm', 'header': {'msg_type': 'comm'}, 'buffers': [b'123'], 'content': {'comm_id': 'foobar', 'data': {'state': {'foo': 'bar'}, 'buffer_paths': [['path']]}}})
def test_widget_comm_buffer_message_single(self, executor, cell_mock, message_mock):
    executor.execute_cell(cell_mock, 0)
    assert message_mock.call_count == 2
    assert executor.widget_state == {'foobar': {'foo': 'bar'}}
    assert executor.widget_buffers == {'foobar': {('path',): {'data': 'MTIz', 'encoding': 'base64', 'path': ['path']}}}
    assert cell_mock.outputs == []