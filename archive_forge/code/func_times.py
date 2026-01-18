from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.test.combinations.times', v1=[])
def times(*combined):
    """Generate a product of N sets of combinations.

  times(combine(a=[1,2]), combine(b=[3,4])) == combine(a=[1,2], b=[3,4])

  Args:
    *combined: N lists of dictionaries that specify combinations.

  Returns:
    a list of dictionaries for each combination.

  Raises:
    ValueError: if some of the inputs have overlapping keys.
  """
    assert combined
    if len(combined) == 1:
        return combined[0]
    first = combined[0]
    rest_combined = times(*combined[1:])
    combined_results = []
    for a in first:
        for b in rest_combined:
            if set(a.keys()).intersection(set(b.keys())):
                raise ValueError('Keys need to not overlap: {} vs {}'.format(a.keys(), b.keys()))
            combined_results.append(OrderedDict(list(a.items()) + list(b.items())))
    return combined_results