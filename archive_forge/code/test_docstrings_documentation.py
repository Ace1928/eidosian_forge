import re
from inspect import signature
from typing import Optional
import pytest
from sklearn.experimental import (
from sklearn.utils.discovery import all_displays, all_estimators, all_functions
Check function docstrings using numpydoc.