from typing import TYPE_CHECKING
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
def _determine_transform_to_use(self) -> BatchFormat:
    return self.preprocessors[0]._determine_transform_to_use()