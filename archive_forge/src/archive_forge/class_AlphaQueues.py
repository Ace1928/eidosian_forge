from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
class AlphaQueues(BaseQueues):
    """Client for queues service in the Cloud Tasks API."""

    def Create(self, parent_ref, queue_ref, retry_config=None, rate_limits=None, pull_target=None, app_engine_http_target=None, http_target=None):
        """Prepares and sends a Create request for creating a queue."""
        targets = (app_engine_http_target, http_target)
        if sum([1 if x is not None else 0 for x in targets]) > 1:
            raise CreatingHttpAndAppEngineQueueError('Attempting to send multiple queue target types simultaneously: {} , {}'.format(six.text_type(app_engine_http_target), six.text_type(http_target)))
        targets = (pull_target, app_engine_http_target, http_target)
        if sum([1 if x is not None else 0 for x in targets]) > 1:
            raise CreatingPullAndAppEngineQueueError('Attempting to send multiple queue target types simultaneously')
        queue = self.messages.Queue(name=queue_ref.RelativeName(), retryConfig=retry_config, rateLimits=rate_limits, pullTarget=pull_target, appEngineHttpTarget=app_engine_http_target, httpTarget=http_target)
        request = self.messages.CloudtasksProjectsLocationsQueuesCreateRequest(parent=parent_ref.RelativeName(), queue=queue)
        return self.queues_service.Create(request)

    def Patch(self, queue_ref, updated_fields, retry_config=None, rate_limits=None, app_engine_routing_override=None, http_uri_override=None, http_method_override=None, http_header_override=None, http_oauth_email_override=None, http_oauth_scope_override=None, http_oidc_email_override=None, http_oidc_audience_override=None):
        """Prepares and sends a Patch request for modifying a queue."""
        if not any([retry_config, rate_limits]):
            if _NeitherUpdateNorClear([app_engine_routing_override], ['appEngineRoutingOverride'], updated_fields) and _NeitherUpdateNorClear([http_uri_override, http_method_override, http_header_override, http_oauth_email_override, http_oauth_scope_override, http_oidc_email_override, http_oidc_audience_override], http_target_update_masks_list, updated_fields):
                raise NoFieldsSpecifiedError('Must specify at least one field to update.')
        queue = self.messages.Queue(name=queue_ref.RelativeName())
        if retry_config is not None:
            queue.retryConfig = retry_config
        if rate_limits is not None:
            queue.rateLimits = rate_limits
        if app_engine_routing_override is not None:
            if _IsEmptyConfig(app_engine_routing_override):
                queue.appEngineHttpTarget = self.messages.AppEngineHttpTarget()
            else:
                queue.appEngineHttpTarget = self.messages.AppEngineHttpTarget(appEngineRoutingOverride=app_engine_routing_override)
        _GenerateHttpTargetUpdateMask(self.messages, queue, updated_fields, http_uri_override, http_method_override, http_header_override, http_oauth_email_override, http_oauth_scope_override, http_oidc_email_override, http_oidc_audience_override)
        update_mask = ','.join(updated_fields)
        request = self.messages.CloudtasksProjectsLocationsQueuesPatchRequest(name=queue_ref.RelativeName(), queue=queue, updateMask=update_mask)
        return self.queues_service.Patch(request)