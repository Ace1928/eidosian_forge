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
class TestConfig1(TestConfigurable):

    def initialize(self, pos_arg=None, a=None):
        self.a = a
        self.pos_arg = pos_arg