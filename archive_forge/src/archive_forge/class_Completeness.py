from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Completeness(_messages.Message):
    """Indicates that the builder claims certain fields in this message to be
  complete.

  Fields:
    arguments: If true, the builder claims that recipe.arguments is complete,
      meaning that all external inputs are properly captured in the recipe.
    environment: If true, the builder claims that recipe.environment is
      claimed to be complete.
    materials: If true, the builder claims that materials are complete,
      usually through some controls to prevent network access. Sometimes
      called "hermetic".
  """
    arguments = _messages.BooleanField(1)
    environment = _messages.BooleanField(2)
    materials = _messages.BooleanField(3)