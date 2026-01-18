import os
import sys
from ...interfaces.base import CommandLine
from .sgegraph import SGEGraphPlugin
from .base import logger
Execute using PBS/Torque

    The plugin_args input to run can be used to control the SGE execution.
    Currently supported options are:

    - template : template to use for batch job submission
    - qsub_args : arguments to be prepended to the job execution script in the
                  qsub call

    