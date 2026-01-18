import collections
import numpy as np
import tensorflow.compat.v2 as tf
def validate_accumulator_extract(self, combiner, data, expected):
    """Validate that the expected results of computing and extracting."""
    acc = combiner.compute(data)
    extracted_data = combiner.extract(acc)
    self.assertAllCloseOrEqual(expected, extracted_data)