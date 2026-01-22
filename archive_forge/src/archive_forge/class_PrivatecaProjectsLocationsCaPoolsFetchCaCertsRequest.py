from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsFetchCaCertsRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsFetchCaCertsRequest object.

  Fields:
    caPool: Required. The resource name for the CaPool in the format
      `projects/*/locations/*/caPools/*`.
    fetchCaCertsRequest: A FetchCaCertsRequest resource to be passed as the
      request body.
  """
    caPool = _messages.StringField(1, required=True)
    fetchCaCertsRequest = _messages.MessageField('FetchCaCertsRequest', 2)