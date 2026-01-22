from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
def GetComponents(runtimes):
    """Gets a list of required components.

  Args:
    runtimes: A list containing the required runtime ids.
  Returns:
    A list of components that must be present.
  """
    components = ['app-engine-python']
    for requested_runtime in runtimes:
        for component_runtime, component in six.iteritems(_RUNTIME_COMPONENTS):
            if component_runtime in requested_runtime:
                components.append(component)
    return components