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
class BaseParent:

    def method(self):
        """ordinary method (base)"""

    def base_method(self):
        """ordinary method in base only"""

    @property
    def prop(self):
        """property"""

    @staticmethod
    def static():
        """static method"""

    @classmethod
    def clsmtd(cls):
        """class method"""