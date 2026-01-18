import pathlib
import pickle
import unittest
from traits.testing.optional_dependencies import (

    Iterate over the pickle files in the test_data directory.

    Skip files that correspond to a protocol not supported with
    the current version of Python.

    Yields paths to pickle files.
    