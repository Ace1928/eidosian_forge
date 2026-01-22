from typing import TYPE_CHECKING
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
def _fit(self, ds: Dataset) -> Preprocessor:
    for preprocessor in self.preprocessors[:-1]:
        ds = preprocessor.fit_transform(ds)
    self.preprocessors[-1].fit(ds)
    return self