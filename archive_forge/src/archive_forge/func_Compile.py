from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
from apitools.base.protorpclite import messages as protorpc_message
from apitools.base.py import encoding as protorpc_encoding
from googlecloudsdk.core.resource import resource_projection_parser
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
from six.moves import range  # pylint: disable=redefined-builtin
def Compile(expression='', defaults=None, symbols=None, by_columns=False, retain_none_values=False):
    """Compiles a resource projection expression.

  Args:
    expression: The resource projection string.
    defaults: resource_projection_spec.ProjectionSpec defaults.
    symbols: Transform function symbol table dict indexed by function name.
    by_columns: Project to a list of columns if True.
    retain_none_values: Retain dict entries with None values.

  Returns:
    A Projector containing the compiled expression ready for Evaluate().
  """
    projection = resource_projection_parser.Parse(expression, defaults=defaults, symbols=symbols, compiler=Compile)
    return Projector(projection, by_columns=by_columns, retain_none_values=retain_none_values)