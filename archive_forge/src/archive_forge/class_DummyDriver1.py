import sys
from unittest.mock import Mock
from libcloud.test import unittest
from libcloud.common.base import BaseDriver
class DummyDriver1(BaseDriver):
    pass