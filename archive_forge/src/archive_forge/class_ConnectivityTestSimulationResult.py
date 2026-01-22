from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectivityTestSimulationResult(_messages.Message):
    """ConnectivityTestSimulationResult contains results for a single
  connectivity test from two network configurations, i.e. original and
  proposed network configurations.

  Fields:
    baseConfigResult: Reachability details (result+traces) for the base
      config.
    destination: Destination endpoint.
    proposedConfigResult: Reachability details (result+traces) for the
      proposed config.
    protocol: Protocol name.
    resultsDiffer: Whether base and proposed config results are different.
    source: Source endpoint.
    testUri: Full resource path (i.e. uri) of the connectivity test using the
      form: 'projects/{project}/locations/{location}/connectivityTestSimulatio
      nResults/{result}'
  """
    baseConfigResult = _messages.MessageField('ReachabilityDetails', 1)
    destination = _messages.MessageField('Endpoint', 2)
    proposedConfigResult = _messages.MessageField('ReachabilityDetails', 3)
    protocol = _messages.StringField(4)
    resultsDiffer = _messages.BooleanField(5)
    source = _messages.MessageField('Endpoint', 6)
    testUri = _messages.StringField(7)