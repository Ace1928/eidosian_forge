import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
class StepStatsAnalysis(collections.namedtuple('StepStatsAnalysis', ('chrome_trace', 'allocator_maximums'))):
    """Stores the step stats analysis output.

  Parameters:
    chrome_trace: A dict containing the chrome trace analysis.
    allocator_maximums: A dict mapping allocator names to AllocationMaximum.
  """
    pass