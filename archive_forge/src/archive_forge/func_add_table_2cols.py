from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def add_table_2cols(self, res, title=None, gleft=None, gright=None, yname=None, xname=None):
    """
        Add a double table, 2 tables with one column merged horizontally

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        title : str, optional
            if None, then a default title is used.
        gleft : list[tuple], optional
            elements for the left table, tuples are (name, value) pairs
            If gleft is None, then a default table is created
        gright : list[tuple], optional
            elements for the right table, tuples are (name, value) pairs
        yname : str, optional
            optional name for the endogenous variable, default is "y"
        xname : list[str], optional
            optional names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model.
        """
    table = summary_top(res, title=title, gleft=gleft, gright=gright, yname=yname, xname=xname)
    self.tables.append(table)