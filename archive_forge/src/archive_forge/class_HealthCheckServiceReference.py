from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthCheckServiceReference(_messages.Message):
    """A full or valid partial URL to a health check service. For example, the
  following are valid URLs: -
  https://www.googleapis.com/compute/beta/projects/project-id/regions/us-
  west1/healthCheckServices/health-check-service - projects/project-
  id/regions/us-west1/healthCheckServices/health-check-service - regions/us-
  west1/healthCheckServices/health-check-service

  Fields:
    healthCheckService: A string attribute.
  """
    healthCheckService = _messages.StringField(1)