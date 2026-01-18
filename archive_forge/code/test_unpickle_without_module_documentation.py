import unittest
import pickle
import sys
import tempfile
from pathlib import Path
Create a module that uses Numba, import a function from it.
        Then delete the module and pickle the function. The function
        should load from the pickle without a problem.

        Note - This is a simplified version of how Numba might be used
        on a distributed system using e.g. dask distributed. With the
        pickle being sent to the worker but not the original module.
        