from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ServiceContext(_messages.Message):
    """Describes a running service that sends errors. Its version changes over
  time and multiple versions can run in parallel.

  Fields:
    resourceType: Type of the MonitoredResource. List of possible values:
      https://cloud.google.com/monitoring/api/resources Value is set
      automatically for incoming errors and must not be set when reporting
      errors.
    service: An identifier of the service, such as the name of the executable,
      job, or Google App Engine service name. This field is expected to have a
      low number of values that are relatively stable over time, as opposed to
      `version`, which can be changed whenever new code is deployed. Contains
      the service name for error reports extracted from Google App Engine logs
      or `default` if the App Engine default service is used.
    version: Represents the source code version that the developer provided,
      which could represent a version label or a Git SHA-1 hash, for example.
      For App Engine standard environment, the version is set to the version
      of the app.
  """
    resourceType = _messages.StringField(1)
    service = _messages.StringField(2)
    version = _messages.StringField(3)