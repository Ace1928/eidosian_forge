from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebServerConfig(_messages.Message):
    """The configuration settings for the Airflow web server App Engine
  instance. Supported for Cloud Composer environments in versions
  composer-1.*.*-airflow-*.*.*.

  Fields:
    machineType: Optional. Machine type on which Airflow web server is
      running. It has to be one of: composer-n1-webserver-2,
      composer-n1-webserver-4 or composer-n1-webserver-8. If not specified,
      composer-n1-webserver-2 will be used. Value custom is returned only in
      response, if Airflow web server parameters were manually changed to a
      non-standard values.
  """
    machineType = _messages.StringField(1)