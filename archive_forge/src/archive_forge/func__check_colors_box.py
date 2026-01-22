import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def _check_colors_box(bp, box_c, whiskers_c, medians_c, caps_c='k', fliers_c=None):
    if fliers_c is None:
        fliers_c = 'k'
    _check_colors(bp['boxes'], linecolors=[box_c] * len(bp['boxes']))
    _check_colors(bp['whiskers'], linecolors=[whiskers_c] * len(bp['whiskers']))
    _check_colors(bp['medians'], linecolors=[medians_c] * len(bp['medians']))
    _check_colors(bp['fliers'], linecolors=[fliers_c] * len(bp['fliers']))
    _check_colors(bp['caps'], linecolors=[caps_c] * len(bp['caps']))