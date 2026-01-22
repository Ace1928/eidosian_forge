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
def _get_function_source_definition(func: Callable) -> str:
    """Get the source code of a function.

    This function is modified from KFP.  The original source is below:
    https://github.com/kubeflow/pipelines/blob/b6406b02f45cdb195c7b99e2f6d22bf85b12268b/sdk/python/kfp/components/_python_op.py#L300-L319.
    """
    func_code = inspect.getsource(func)
    func_code = textwrap.dedent(func_code)
    func_code_lines = func_code.split('\n')
    func_code_lines = itertools.dropwhile(lambda x: not (x.startswith('def') or x.startswith('@wandb_log')), func_code_lines)
    if not func_code_lines:
        raise ValueError(f'Failed to dedent and clean up the source of function "{func.__name__}". It is probably not properly indented.')
    return '\n'.join(func_code_lines)