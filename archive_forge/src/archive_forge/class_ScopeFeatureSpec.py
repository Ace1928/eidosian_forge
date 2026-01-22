from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScopeFeatureSpec(_messages.Message):
    """ScopeFeatureSpec contains feature specs for a fleet scope.

  Fields:
    helloworld: Spec for the HelloWorld feature at the scope level
  """
    helloworld = _messages.MessageField('HelloWorldScopeSpec', 1)