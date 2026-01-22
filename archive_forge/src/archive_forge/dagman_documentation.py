import os
import sys
import uuid
import time
from warnings import warn
from .base import GraphPluginBase, logger
from ...interfaces.base import CommandLine
Execute using Condor DAGMan

    The plugin_args input to run can be used to control the DAGMan execution.
    The value of most arguments can be a literal string or a filename, where in
    the latter case the content of the file will be used as the argument value.

    Currently supported options are:

    - submit_template : submit spec template for individual jobs in a DAG (see
                 CondorDAGManPlugin.default_submit_template for the default.
    - initial_specs : additional submit specs that are prepended to any job's
                 submit file
    - override_specs : additional submit specs that are appended to any job's
                 submit file
    - wrapper_cmd : path to an executable that will be started instead of a node
                 script. This is useful for wrapper script that execute certain
                 functionality prior or after a node runs. If this option is
                 given the wrapper command is called with the respective Python
                 executable and the path to the node script as final arguments
    - wrapper_args : optional additional arguments to a wrapper command
    - dagman_args : arguments to be prepended to the arguments of the
                    condor_submit_dag call
    - block : if True the plugin call will block until Condor has finished
                 processing the entire workflow (default: False)
    