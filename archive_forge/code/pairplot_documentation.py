import warnings
from copy import deepcopy
from uuid import uuid4
import bokeh.plotting as bkp
import numpy as np
from bokeh.models import CDSView, ColumnDataSource, GroupFilter, Span
from ....rcparams import rcParams
from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults
Compute subplots dimensions for two or more variables.