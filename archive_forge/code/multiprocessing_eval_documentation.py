import torch
import random
import os
import signal
import parlai.utils.distributed as distributed_utils
import parlai.scripts.eval_model as eval_model
from parlai.core.script import ParlaiScript, register_script

    Perform a fork() to many processes.
    