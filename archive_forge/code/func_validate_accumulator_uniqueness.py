import collections
import numpy as np
import tensorflow.compat.v2 as tf
def validate_accumulator_uniqueness(self, combiner, data):
    """Validate that every call to compute creates a unique accumulator."""
    acc = combiner.compute(data)
    acc2 = combiner.compute(data)
    self.assertIsNot(acc, acc2)
    self.compare_accumulators(acc, acc2)