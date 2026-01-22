import sys
from google.protobuf import message
from tensorflow.core.profiler import tfprof_options_pb2
from tensorflow.core.profiler import tfprof_output_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util.tf_export import tf_export
def _build_advisor_options(options):
    """Build tfprof.AdvisorOptionsProto.

  Args:
    options: A dictionary of options. See ALL_ADVICE example.

  Returns:
    tfprof.AdvisorOptionsProto.
  """
    opts = tfprof_options_pb2.AdvisorOptionsProto()
    if options is None:
        return opts
    for checker, checker_opts in options.items():
        checker_ops_pb = tfprof_options_pb2.AdvisorOptionsProto.CheckerOption()
        for k, v in checker_opts.items():
            checker_ops_pb[k] = v
        opts.checkers[checker].MergeFrom(checker_ops_pb)
    return opts