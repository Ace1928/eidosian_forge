from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distribute.experimental.CentralStorageStrategy'])
class CentralStorageStrategyV1(distribute_lib.StrategyV1):
    __doc__ = CentralStorageStrategy.__doc__

    def __init__(self, compute_devices=None, parameter_device=None):
        super(CentralStorageStrategyV1, self).__init__(parameter_server_strategy.ParameterServerStrategyExtended(self, compute_devices=compute_devices, parameter_device=parameter_device))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('CentralStorageStrategy')
    __init__.__doc__ = CentralStorageStrategy.__init__.__doc__