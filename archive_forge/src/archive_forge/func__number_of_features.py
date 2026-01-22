from typing import List, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def _number_of_features(self) -> int:
    return sum(self.embedding_dimension) + self.num_dynamic_real_features + self.num_time_features + self.num_static_real_features + self.input_size * 2