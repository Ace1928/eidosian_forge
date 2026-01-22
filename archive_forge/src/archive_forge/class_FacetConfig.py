from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FacetConfig(_messages.Message):
    """FacetConfig allows for configuration of faceted search.

  Fields:
    maxValues: Maximum number of facet values to return in a facet. Default is
      10.
    operators: The list of search operators to include in a facet.
  """
    maxValues = _messages.IntegerField(1)
    operators = _messages.StringField(2, repeated=True)