from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsLocationPrometheusApiV1QueryExemplarsRequest(_messages.Message):
    """A MonitoringProjectsLocationPrometheusApiV1QueryExemplarsRequest object.

  Fields:
    location: Location of the resource information. Has to be "global" now.
    name: The project on which to execute the request. Data associcated with
      the project's workspace stored under the The format is:
      projects/PROJECT_ID_OR_NUMBER. Open source API but used as a request
      path prefix to distinguish different virtual Prometheus instances of
      Google Prometheus Engine.
    queryExemplarsRequest: A QueryExemplarsRequest resource to be passed as
      the request body.
  """
    location = _messages.StringField(1, required=True)
    name = _messages.StringField(2, required=True)
    queryExemplarsRequest = _messages.MessageField('QueryExemplarsRequest', 3)