from typing import Dict
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
class HasContribPredictionCol(Params):
    """
    Mixin for param pred_contrib_col: contribution prediction column name.

    Output is a 3-dim array, with (rows, groups, columns + 1) for classification case.
    Else, it can be a 2 dimension for regression case.
    """
    pred_contrib_col: 'Param[str]' = Param(Params._dummy(), 'pred_contrib_col', 'feature contributions to individual predictions.', typeConverter=TypeConverters.toString)