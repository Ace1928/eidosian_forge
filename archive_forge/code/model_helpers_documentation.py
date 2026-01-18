import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Type, TypeVar
from lightning_utilities.core.imports import RequirementCache
from torch import nn
from typing_extensions import Concatenate, ParamSpec
import pytorch_lightning as pl
Drop-in replacement for @classmethod, but raises an exception when the decorated method is called on an instance
    instead of a class type.