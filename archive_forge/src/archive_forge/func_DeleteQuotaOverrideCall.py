from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def DeleteQuotaOverrideCall(service, consumer, metric, unit, override_id, force=False):
    """Delete a quota override.

  Args:
    service: The service to delete a quota aoverride for.
    consumer: The consumer to delete a quota override for, e.g. "projects/123".
    metric: The quota metric name.
    unit: The unit of quota metric.
    override_id: The override ID.
    force: Force override deletion even if the change results in a substantial
      decrease in available quota.

  Raises:
    exceptions.DeleteQuotaOverridePermissionDeniedException: when deleting an
    override fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The quota override operation.
  """
    _ValidateConsumer(consumer)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    parent = _GetMetricResourceName(service, consumer, metric, unit)
    name = _LIMIT_OVERRIDE_RESOURCE % (parent, override_id)
    request = messages.ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesDeleteRequest(name=name, force=force)
    try:
        return client.services_consumerQuotaMetrics_limits_producerOverrides.Delete(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.DeleteQuotaOverridePermissionDeniedException)