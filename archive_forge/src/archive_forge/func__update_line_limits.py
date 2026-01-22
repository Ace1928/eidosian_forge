from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def _update_line_limits(self, line):
    """
        Figures out the data limit of the given line, updating self.dataLim.
        """
    path = line.get_path()
    if path.vertices.size == 0:
        return
    line_trf = line.get_transform()
    if line_trf == self.transData:
        data_path = path
    elif any(line_trf.contains_branch_seperately(self.transData)):
        trf_to_data = line_trf - self.transData
        if self.transData.is_affine:
            line_trans_path = line._get_transformed_path()
            na_path, _ = line_trans_path.get_transformed_path_and_affine()
            data_path = trf_to_data.transform_path_affine(na_path)
        else:
            data_path = trf_to_data.transform_path(path)
    else:
        data_path = path
    if not data_path.vertices.size:
        return
    updatex, updatey = line_trf.contains_branch_seperately(self.transData)
    if self.name != 'rectilinear':
        if updatex and line_trf == self.get_yaxis_transform():
            updatex = False
        if updatey and line_trf == self.get_xaxis_transform():
            updatey = False
    self.dataLim.update_from_path(data_path, self.ignore_existing_data_limits, updatex=updatex, updatey=updatey)
    self.ignore_existing_data_limits = False