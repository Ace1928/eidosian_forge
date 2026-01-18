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
@pytest.mark.skipif(int(version_info[0]) < 7, reason='requires client 7+')
def test_start_new_kernel_client_cleans_up_kernel_on_failure():

    class FakeClient(KernelClient):

        def start_channels(self, shell: bool=True, iopub: bool=True, stdin: bool=True, hb: bool=True, control: bool=True) -> None:
            raise Exception('Any error')

        def stop_channels(self) -> None:
            pass
    nb = nbformat.v4.new_notebook()
    km = KernelManager()
    km.client_factory = FakeClient
    executor = NotebookClient(nb, km=km)
    executor.start_new_kernel()
    assert km.has_kernel
    assert executor.km is not None
    with pytest.raises(Exception) as err:
        executor.start_new_kernel_client()
    assert str(err.value.args[0]) == 'Any error'
    assert executor.kc is None
    assert executor.km is None
    assert not km.has_kernel