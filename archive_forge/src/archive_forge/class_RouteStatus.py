from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteStatus(_messages.Message):
    """RouteStatus communicates the observed state of the Route (from the
  controller).

  Fields:
    address: Similar to url, information on where the service is available on
      HTTP.
    conditions: Conditions communicates information about ongoing/complete
      reconciliation processes that bring the "spec" inline with the observed
      state of the world.
    observedGeneration: ObservedGeneration is the 'Generation' of the Route
      that was last processed by the controller. Clients polling for completed
      reconciliation should poll until observedGeneration =
      metadata.generation and the Ready condition's status is True or False.
      Note that providing a TrafficTarget that has latest_revision=True will
      result in a Route that does not increment either its metadata.generation
      or its observedGeneration, as new "latest ready" revisions from the
      Configuration are processed without an update to the Route's spec.
    traffic: Traffic holds the configured traffic distribution. These entries
      will always contain RevisionName references. When ConfigurationName
      appears in the spec, this will hold the LatestReadyRevisionName that was
      last observed.
    url: URL holds the url that will distribute traffic over the provided
      traffic targets. It generally has the form: `https://{route-
      hash}-{project-hash}-{cluster-level-suffix}.a.run.app`
  """
    address = _messages.MessageField('Addressable', 1)
    conditions = _messages.MessageField('GoogleCloudRunV1Condition', 2, repeated=True)
    observedGeneration = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    traffic = _messages.MessageField('TrafficTarget', 4, repeated=True)
    url = _messages.StringField(5)