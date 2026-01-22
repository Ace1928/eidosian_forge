import contextlib
import copy
import difflib
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import (
import numpy as np
from gym.wrappers import (
from gym.wrappers.compatibility import EnvCompatibility
from gym.wrappers.env_checker import PassiveEnvChecker
from gym import Env, error, logger
def _check_namespace_exists(ns: Optional[str]):
    """Check if a namespace exists. If it doesn't, print a helpful error message."""
    if ns is None:
        return
    namespaces = {spec_.namespace for spec_ in registry.values() if spec_.namespace is not None}
    if ns in namespaces:
        return
    suggestion = difflib.get_close_matches(ns, namespaces, n=1) if len(namespaces) > 0 else None
    suggestion_msg = f'Did you mean: `{suggestion[0]}`?' if suggestion else f'Have you installed the proper package for {ns}?'
    raise error.NamespaceNotFound(f'Namespace {ns} not found. {suggestion_msg}')