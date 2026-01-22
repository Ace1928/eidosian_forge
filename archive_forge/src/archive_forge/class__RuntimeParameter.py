from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
class _RuntimeParameter(Parameter):
    """A runtime Parameter.

  Attributes:
    aggregator: True if parameter is an aggregator (not aggregated by updater).
    generate: True if values must be generated for this parameter.
    updater_class: The updater class.
    value: A default value from the program state.
  """

    def __init__(self, parameter, updater_class, value, aggregator):
        super(_RuntimeParameter, self).__init__(parameter.column, name=parameter.name)
        self.generate = False
        self.updater_class = updater_class
        self.value = value
        self.aggregator = aggregator