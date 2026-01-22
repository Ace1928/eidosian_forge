import unittest
from warnings import catch_warnings
from unittest.test.testmock.support import is_instance
from unittest.mock import MagicMock, Mock, patch, sentinel, mock_open, call
class SampleException(Exception):
    pass