import collections
import numpy as np
import tensorflow.compat.v2 as tf
def validate_accumulator_serialize_and_deserialize(self, combiner, data, expected):
    """Validate that the serialize<->deserialize loop loses no data."""
    acc = combiner.compute(data)
    serialized_data = combiner.serialize(acc)
    deserialized_data = combiner.deserialize(serialized_data)
    self.compare_accumulators(acc, deserialized_data)
    self.compare_accumulators(expected, deserialized_data)