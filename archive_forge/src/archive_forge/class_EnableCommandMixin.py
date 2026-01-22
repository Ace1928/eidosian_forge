from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
class EnableCommandMixin(FeatureCommand):
    """A mixin for functionality to enable a Feature."""

    def Enable(self, feature):
        project = properties.VALUES.core.project.GetOrFail()
        if self.feature.api:
            enable_api.EnableServiceIfDisabled(project, self.feature.api)
        parent = util.LocationResourceName(project)
        try:
            retryer = retry.Retryer(max_retrials=4, exponential_sleep_multiplier=1.75)
            op = retryer.RetryOnException(self.hubclient.CreateFeature, args=(parent, self.feature_name, feature), should_retry_if=self._FeatureAPINotEnabled, sleep_ms=1000)
        except retry.MaxRetrialsException:
            raise exceptions.Error('Retry limit exceeded waiting for {} to enable'.format(self.feature.display_name))
        except apitools_exceptions.HttpConflictError as e:
            error = core_api_exceptions.HttpErrorPayload(e)
            if error.status_description != 'ALREADY_EXISTS':
                raise
            log.status.Print('{} Feature for project [{}] is already enabled'.format(self.feature.display_name, project))
            return
        msg = 'Waiting for Feature {} to be created'.format(self.feature.display_name)
        return self.WaitForHubOp(self.hubclient.feature_waiter, op=op, message=msg)

    def _FeatureAPINotEnabled(self, exc_type, exc_value, traceback, state):
        del traceback, state
        if not self.feature.api:
            return False
        if exc_type != apitools_exceptions.HttpBadRequestError:
            return False
        error = core_api_exceptions.HttpErrorPayload(exc_value)
        if not (error.status_description == 'FAILED_PRECONDITION' and self.feature.api in error.message and ('is not enabled' in error.message)):
            return False
        log.status.Print('Waiting for service API enablement to finish...')
        return True