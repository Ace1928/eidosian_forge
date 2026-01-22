import sys
from unittest import mock
import types
import warnings
import unittest
import os
import subprocess
import threading
from numba import config, njit
from numba.tests.support import TestCase
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
@numba.extending.register_model(DummyType)
class DummyModel(numba.extending.models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('value', numba.types.float64)]
        super(DummyModel, self).__init__(dmm, fe_type, members)