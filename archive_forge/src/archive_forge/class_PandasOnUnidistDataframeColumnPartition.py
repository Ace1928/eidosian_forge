import warnings
import pandas
import unidist
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.core.execution.unidist.common import UnidistWrapper
from modin.core.execution.unidist.common.utils import deserialize
from modin.utils import _inherit_docstrings
from .partition import PandasOnUnidistDataframePartition
@_inherit_docstrings(PandasOnUnidistDataframeVirtualPartition.__init__)
class PandasOnUnidistDataframeColumnPartition(PandasOnUnidistDataframeVirtualPartition):
    axis = 0