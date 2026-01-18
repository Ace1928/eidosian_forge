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
def test_warns_that_defaulting_to_pandas():
    with warns_that_defaulting_to_pandas():
        ErrorMessage.default_to_pandas()
    with warns_that_defaulting_to_pandas():
        ErrorMessage.default_to_pandas(message='Function name')