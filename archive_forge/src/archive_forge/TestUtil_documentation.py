import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
Constructor.

        :param needs_module: a callable taking a module name as a
            parameter returing True if the module should be loaded.
        