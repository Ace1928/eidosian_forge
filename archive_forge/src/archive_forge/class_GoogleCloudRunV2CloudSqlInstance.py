from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2CloudSqlInstance(_messages.Message):
    """Represents a set of Cloud SQL instances. Each one will be available
  under /cloudsql/[instance]. Visit
  https://cloud.google.com/sql/docs/mysql/connect-run for more information on
  how to connect Cloud SQL and Cloud Run.

  Fields:
    instances: The Cloud SQL instance connection names, as can be found in
      https://console.cloud.google.com/sql/instances. Visit
      https://cloud.google.com/sql/docs/mysql/connect-run for more information
      on how to connect Cloud SQL and Cloud Run. Format:
      {project}:{location}:{instance}
  """
    instances = _messages.StringField(1, repeated=True)