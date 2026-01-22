from typing import Dict
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
class HasEnableSparseDataOptim(Params):
    """
    This is a Params based class that is extended by _SparkXGBParams
    and holds the variable to store the boolean config of enabling sparse data optimization.
    """
    enable_sparse_data_optim = Param(Params._dummy(), 'enable_sparse_data_optim', 'This stores the boolean config of enabling sparse data optimization, if enabled, Xgboost DMatrix object will be constructed from sparse matrix instead of dense matrix. This config is disabled by default. If most of examples in your training dataset contains sparse features, we suggest to enable this config.', typeConverter=TypeConverters.toBoolean)

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(enable_sparse_data_optim=False)