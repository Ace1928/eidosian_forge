from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
Test header column, spacer, first line and last line in verbose mode.