from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectPipeline(RefreshMixin, ObjectDeleteMixin, RESTObject):
    bridges: 'ProjectPipelineBridgeManager'
    jobs: 'ProjectPipelineJobManager'
    test_report: 'ProjectPipelineTestReportManager'
    test_report_summary: 'ProjectPipelineTestReportSummaryManager'
    variables: 'ProjectPipelineVariableManager'

    @cli.register_custom_action('ProjectPipeline')
    @exc.on_http_error(exc.GitlabPipelineCancelError)
    def cancel(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Cancel the job.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabPipelineCancelError: If the request failed
        """
        path = f'{self.manager.path}/{self.encoded_id}/cancel'
        return self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('ProjectPipeline')
    @exc.on_http_error(exc.GitlabPipelineRetryError)
    def retry(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Retry the job.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabPipelineRetryError: If the request failed
        """
        path = f'{self.manager.path}/{self.encoded_id}/retry'
        return self.manager.gitlab.http_post(path, **kwargs)