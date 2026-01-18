import inspect
import itertools
import textwrap
from typing import Callable, List, Mapping, Optional
import wandb
from .wandb_logging import wandb_log
import typing
from typing import NamedTuple
import collections
from collections import namedtuple
import kfp
from kfp import components
from kfp.components import InputPath, OutputPath
import wandb
def unpatch_kfp():
    unpatch('kfp.components')
    unpatch('kfp.components._python_op')
    unpatch('wandb.integration.kfp')