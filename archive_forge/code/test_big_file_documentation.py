import contextlib
import errno
import os
import resource
import sys
from breezy import osutils, tests
from breezy.tests import features, script
Tests with "big" files.

These are meant to ensure that Breezy never keeps full copies of files in
memory.
