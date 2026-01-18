import json
from textwrap import dedent, indent
from unittest.mock import Mock, patch
import numpy as np
import pandas
import pytest
import modin.pandas as pd
import modin.utils
from modin.error_message import ErrorMessage
from modin.tests.pandas.utils import create_test_dfs
@pytest.fixture(scope='module')
def wrapped_cls():

    @modin.utils._inherit_docstrings(BaseChild)
    class Wrapped:

        def method(self):
            pass

        def base_method(self):
            pass

        def own_method(self):
            pass

        def no_overwrite(self):
            """not overwritten doc"""

        @property
        def prop(self):
            return None

        @staticmethod
        def static():
            pass

        @classmethod
        def clsmtd(cls):
            pass
        F = property(method)
    return Wrapped