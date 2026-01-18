import sys
import warnings
from functools import partial
from textwrap import indent
import pytest
from nibabel.deprecator import (
from ..testing import clear_and_catch_warnings
Test deprecator class creation with custom warnings and errors