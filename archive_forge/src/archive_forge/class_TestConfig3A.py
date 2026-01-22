from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
class TestConfig3A(TestConfig3):

    def initialize(self, a=None):
        self.a = a