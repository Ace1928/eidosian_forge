import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale

    Test that string representations of long-double roundtrip both
    for array casting and scalar coercion, see also gh-15608.
    