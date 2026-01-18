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
def test_start_new_kernel_history_file_setting():
    nb = nbformat.v4.new_notebook()
    km = KernelManager()
    executor = NotebookClient(nb, km=km)
    kc = km.client()
    assert executor.extra_arguments == []
    executor.start_new_kernel()
    assert executor.extra_arguments == ['--HistoryManager.hist_file=:memory:']
    executor.start_new_kernel()
    assert executor.extra_arguments == ['--HistoryManager.hist_file=:memory:']
    kc.shutdown()
    km.cleanup_resources()
    kc.stop_channels()