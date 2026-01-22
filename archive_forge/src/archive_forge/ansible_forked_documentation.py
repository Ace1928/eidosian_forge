from __future__ import absolute_import, division, print_function
import os
import pickle
import tempfile
import warnings
from pytest import Item, hookimpl
from _pytest.runner import runtestprotocol
Convert a wait status to an exit code.