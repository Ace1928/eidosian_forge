import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def add_distributed_training_args(self):
    """
        Add CLI args for distributed training.
        """
    grp = self.add_argument_group('Distributed Training')
    grp.add_argument('--distributed-world-size', type=int, help='Number of workers.')
    grp.add_argument('--verbose', type='bool', default=False, help='All workers print output.', hidden=True)
    return grp