import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
class HParam:
    """A hyperparameter in an experiment.

    This class describes a hyperparameter in the abstract. It ranges
    over a domain of values, but is not bound to any particular value.
    """

    def __init__(self, name, domain=None, display_name=None, description=None):
        """Create a hyperparameter object.

        Args:
          name: A string ID for this hyperparameter, which should be unique
            within an experiment.
          domain: An optional `Domain` object describing the values that
            this hyperparameter can take on.
          display_name: An optional human-readable display name (`str`).
          description: An optional Markdown string describing this
            hyperparameter.

        Raises:
          ValueError: If `domain` is not a `Domain`.
        """
        self._name = name
        self._domain = domain
        self._display_name = display_name
        self._description = description
        if not isinstance(self._domain, (Domain, type(None))):
            raise ValueError('not a domain: %r' % (self._domain,))

    def __str__(self):
        return '<HParam %r: %s>' % (self._name, self._domain)

    def __repr__(self):
        fields = [('name', self._name), ('domain', self._domain), ('display_name', self._display_name), ('description', self._description)]
        fields_string = ', '.join(('%s=%r' % (k, v) for k, v in fields))
        return 'HParam(%s)' % fields_string

    @property
    def name(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    @property
    def display_name(self):
        return self._display_name

    @property
    def description(self):
        return self._description