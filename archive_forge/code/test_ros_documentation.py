from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros

    Oahu arsenic data from Nondetects and Data Analysis by
    Dennis R. Helsel (John Wiley, 2005)

    Plotting positions are fudged since relative to source data since
    modeled data is what matters and (source data plot positions are
    not uniformly spaced, which seems weird)
    