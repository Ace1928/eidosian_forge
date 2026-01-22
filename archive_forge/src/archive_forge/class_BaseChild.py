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
class BaseChild(BaseParent):
    """this is class docstring"""

    def method(self):
        """ordinary method (child)"""

    def own_method(self):
        """own method"""

    def no_overwrite(self):
        """another own method"""
    F = property(method)