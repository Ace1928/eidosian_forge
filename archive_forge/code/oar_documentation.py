import os
import stat
from time import sleep
import subprocess
import simplejson as json
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
Execute using OAR

    The plugin_args input to run can be used to control the OAR execution.
    Currently supported options are:

    - template : template to use for batch job submission
    - oarsub_args : arguments to be prepended to the job execution
                    script in the oarsub call
    - max_jobname_len: maximum length of the job name.  Default 15.

    