import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
class ConfigPatch(ContextDecorator):

    def __enter__(self):
        assert not prior
        for key in changes.keys():
            prior[key] = config._config[key]
        config._config.update(changes)

    def __exit__(self, exc_type, exc_val, exc_tb):
        config._config.update(prior)
        prior.clear()