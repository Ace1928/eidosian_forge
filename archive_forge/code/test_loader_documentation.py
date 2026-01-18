import functools
import sys
import types
import warnings
import unittest
Helper function for checking that errors in loading are reported.

        :param loader: A loader with some errors.
        :param suite: A suite that should have a late bound error.
        :return: The first error message from the loader and the test object
            from the suite.
        