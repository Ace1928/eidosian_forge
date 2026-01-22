import unittest
import os
import contextlib
import importlib_resources as resources
class StringAnchorMixin:
    anchor01 = 'importlib_resources.tests.data01'
    anchor02 = 'importlib_resources.tests.data02'