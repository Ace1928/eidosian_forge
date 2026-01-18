import operator as op
from numbers import Number
import pytest
import numpy as np
from numpy.polynomial import (
from numpy.testing import (
from numpy.polynomial.polyutils import RankWarning
Test inter-conversion of different polynomial classes.

This tests the convert and cast methods of all the polynomial classes.

