import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
Test excessive stream reads.

        Test that more data can be requested from the stream than available
        without raising an exception.
        