import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
class rc_context:
    """
    Return a context manager for managing rc settings.

    Parameters
    ----------
    rc : dict, optional
        Mapping containing the rcParams to modify temporally.
    fname : str, optional
        Filename of the file containing the rcParams to use inside the rc_context.

    Examples
    --------
    This allows one to do::

        with az.rc_context(fname='pystan.rc'):
            idata = az.load_arviz_data("radon")
            az.plot_posterior(idata, var_names=["gamma"])

    The plot would have settings from 'screen.rc'

    A dictionary can also be passed to the context manager::

        with az.rc_context(rc={'plot.max_subplots': None}, fname='pystan.rc'):
            idata = az.load_arviz_data("radon")
            az.plot_posterior(idata, var_names=["gamma"])

    The 'rc' dictionary takes precedence over the settings loaded from
    'fname'. Passing a dictionary only is also valid.
    """

    def __init__(self, rc=None, fname=None):
        self._orig = rcParams.copy()
        if fname:
            file_rcparams = read_rcfile(fname)
            rcParams.update(file_rcparams)
        if rc:
            rcParams.update(rc)

    def __enter__(self):
        """Define enter method of context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Define exit method of context manager."""
        rcParams.update(self._orig)