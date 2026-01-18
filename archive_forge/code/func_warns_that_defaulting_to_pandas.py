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
def warns_that_defaulting_to_pandas(prefix=None, suffix=None):
    """
    Assert that code warns that it's defaulting to pandas.

    Parameters
    ----------
    prefix : Optional[str]
        If specified, checks that the start of the warning message matches this argument
        before "[Dd]efaulting to pandas".
    suffix : Optional[str]
        If specified, checks that the end of the warning message matches this argument
        after "[Dd]efaulting to pandas".

    Returns
    -------
    pytest.recwarn.WarningsChecker
        A WarningsChecker checking for a UserWarning saying that Modin is
        defaulting to Pandas.
    """
    match = '[Dd]efaulting to pandas'
    if prefix:
        match = match + '(.|\\n)+'
    if suffix:
        match += '(.|\\n)+' + suffix
    return pytest.warns(UserWarning, match=match)