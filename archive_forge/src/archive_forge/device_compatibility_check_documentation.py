import itertools
from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging
Logs a compatibility check if the devices support the policy.

  Currently only logs for the policy mixed_float16. A log is shown only the
  first time this function is called.

  Args:
    policy_name: The name of the dtype policy.
  