import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
class ChunkSize1FifoDiskQueueTest(FifoDiskQueueTest):
    chunksize = 1