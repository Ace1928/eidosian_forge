import numpy as np
import pandas as pd
from .._utils import groupby_apply
from ..doctools import document
from ..exceptions import PlotnineWarning
from ..scales.scale_discrete import scale_discrete
from .binning import fuzzybreaks
from .stat import stat
from .stat_summary import make_summary_fun

            Add `bin` column to each summary result.
            