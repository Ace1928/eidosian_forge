from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
Asking a user to answer yes, no, or cancel.

    :param question: The question asked to the user
    :return: An integer with the answer.
    