from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _GetNestedStatProperty(self, prop_name, nested_prop_name):
    """Gets a nested property name on this object's executionStats.

    Args:
      prop_name: A string of the key name for the outer property on
        executionStats.
      nested_prop_name: A string of the key name of the nested property.

    Returns:
      The string value of the nested property, or None if the outermost
      property or nested property don't exist.
    """
    prop = _GetAdditionalProperty(self.properties.executionStats.additionalProperties, prop_name, '')
    if not prop:
        return None
    nested_prop = _GetAdditionalProperty(prop.object_value.properties, nested_prop_name, '')
    if nested_prop:
        return nested_prop.string_value
    return None