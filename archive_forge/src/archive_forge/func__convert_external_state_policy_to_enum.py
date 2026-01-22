from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _convert_external_state_policy_to_enum(external_state_policy):
    if isinstance(external_state_policy, options_lib.ExternalStatePolicy):
        return external_state_policy
    if external_state_policy == 'warn':
        return options_lib.ExternalStatePolicy.WARN
    if external_state_policy == 'ignore':
        return options_lib.ExternalStatePolicy.IGNORE
    if external_state_policy == 'fail':
        return options_lib.ExternalStatePolicy.FAIL
    raise ValueError(f"Invalid `ExternalStatePolicy.` Supported values include 'warn', 'ignore', and 'fail.' Received {external_state_policy}.")