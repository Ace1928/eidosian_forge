import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
def _floatOrCall(val):
    """Cast the value to float, if that fails call it and then cast.

    This is an "augmented" version of float() to better support
    integration with Pyomo NumericValue objects: if the initial cast to
    float fails by throwing a TypeError (as non-constant NumericValue
    objects will), then it falls back on calling the object and
    returning that value cast to float.

    """
    try:
        return float(val)
    except (TypeError, InvalidValueError):
        pass
    try:
        return float(val())
    except (TypeError, InvalidValueError):
        pass
    try:
        return val.value
    except AttributeError:
        return val