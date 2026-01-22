import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
class RNotReadyError(Exception):
    """Embedded R is not ready to use."""
    pass