import unittest
import os
import sys
import sysconfig
from test.support import (
from distutils.command.config import dump_file, config
from distutils.tests import support
from distutils import log
Tests for distutils.command.config.