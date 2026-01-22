from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_tf_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments
@cached_property
def _setup_strategy(self) -> Tuple['tf.distribute.Strategy', 'tf.distribute.cluster_resolver.TPUClusterResolver']:
    requires_backends(self, ['tf'])
    if self.is_tpu:
        tf.config.experimental_connect_to_cluster(self._setup_tpu)
        tf.tpu.experimental.initialize_tpu_system(self._setup_tpu)
        strategy = tf.distribute.TPUStrategy(self._setup_tpu)
    elif self.is_gpu:
        tf.config.set_visible_devices(self.gpu_list[self.device_idx], 'GPU')
        strategy = tf.distribute.OneDeviceStrategy(device=f'/gpu:{self.device_idx}')
    else:
        tf.config.set_visible_devices([], 'GPU')
        strategy = tf.distribute.OneDeviceStrategy(device=f'/cpu:{self.device_idx}')
    return strategy