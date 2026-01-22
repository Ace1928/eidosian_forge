from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SlsaMetadata(_messages.Message):
    """Other properties of the build.

  Fields:
    buildFinishedOn: The timestamp of when the build completed.
    buildInvocationId: Identifies the particular build invocation, which can
      be useful for finding associated logs or other ad-hoc analysis. The
      value SHOULD be globally unique, per in-toto Provenance spec.
    buildStartedOn: The timestamp of when the build started.
    completeness: Indicates that the builder claims certain fields in this
      message to be complete.
    reproducible: If true, the builder claims that running the recipe on
      materials will produce bit-for-bit identical output.
  """
    buildFinishedOn = _messages.StringField(1)
    buildInvocationId = _messages.StringField(2)
    buildStartedOn = _messages.StringField(3)
    completeness = _messages.MessageField('SlsaCompleteness', 4)
    reproducible = _messages.BooleanField(5)