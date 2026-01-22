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
def advise(self, options):
    """Automatically detect problems and generate reports.

    Args:
      options: A dict of options. See ALL_ADVICE example above.

    Returns:
      An Advise proto that contains the reports from all checkers.
    """
    advise_pb = tfprof_output_pb2.AdviceProto()
    opts = _build_advisor_options(options)
    advise_pb.ParseFromString(print_mdl.Profile('advise'.encode('utf-8'), opts.SerializeToString()))
    return advise_pb