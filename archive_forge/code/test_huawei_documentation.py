import os
import tempfile
from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import huawei
from os_brick.tests.initiator import test_connector
Test the fail disconnect volume case.