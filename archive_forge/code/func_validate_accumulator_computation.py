import collections
import numpy as np
import tensorflow.compat.v2 as tf
def validate_accumulator_computation(self, combiner, data, expected):
    """Validate that various combinations of compute and merge are
        identical."""
    if len(data) < 4:
        raise AssertionError(f'Data must have at least 4 elements. Received len(data)={len(data)}.')
    data_0 = np.array([data[0]])
    data_1 = np.array([data[1]])
    data_2 = np.array(data[2:])
    single_compute = combiner.compute(data)
    all_merge = combiner.merge([combiner.compute(data_0), combiner.compute(data_1), combiner.compute(data_2)])
    self.compare_accumulators(single_compute, all_merge, msg='Sharding data should not change the data output.')
    unordered_all_merge = combiner.merge([combiner.compute(data_1), combiner.compute(data_2), combiner.compute(data_0)])
    self.compare_accumulators(all_merge, unordered_all_merge, msg='The order of merge arguments should not change the data output.')
    hierarchical_merge = combiner.merge([combiner.compute(data_1), combiner.merge([combiner.compute(data_2), combiner.compute(data_0)])])
    self.compare_accumulators(all_merge, hierarchical_merge, msg='Nesting merge arguments should not change the data output.')
    nested_compute = combiner.compute(data_0, combiner.compute(data_1, combiner.compute(data_2)))
    self.compare_accumulators(all_merge, nested_compute, msg='Nesting compute arguments should not change the data output.')
    mixed_compute = combiner.merge([combiner.compute(data_0), combiner.compute(data_1, combiner.compute(data_2))])
    self.compare_accumulators(all_merge, mixed_compute, msg='Mixing merge and compute calls should not change the data output.')
    single_merge = combiner.merge([combiner.merge([combiner.compute(data_0)]), combiner.compute(data_1, combiner.compute(data_2))])
    self.compare_accumulators(all_merge, single_merge, msg='Calling merge with a data length of 1 should not change the data output.')
    self.compare_accumulators(expected, all_merge, msg='Calculated accumulators did not match expected accumulator.')