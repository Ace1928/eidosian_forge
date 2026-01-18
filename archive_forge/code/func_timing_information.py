import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@property
@has_context
def timing_information(self):
    return getattr(flask.g, 'timing_information', {})