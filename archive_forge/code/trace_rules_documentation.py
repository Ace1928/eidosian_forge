import functools
import importlib
import sys
import types
import torch
from .allowed_functions import _disallowed_function_ids, is_user_defined_allowed
from .utils import hashable
from .variables import (

Main entry point for looking up the trace rule (the Dynamo variable) for a given object.
E.g, the lookup result of `torch.amp.autocast_mode.autocast` is `TorchCtxManagerClassVariable`.
