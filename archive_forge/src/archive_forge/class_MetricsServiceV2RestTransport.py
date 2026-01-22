from google.auth.transport.requests import AuthorizedSession  # type: ignore
import json  # type: ignore
import grpc  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import path_template
from google.api_core import gapic_v1
from cloudsdk.google.protobuf import json_format
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging_metrics
from .base import MetricsServiceV2Transport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class MetricsServiceV2RestTransport(MetricsServiceV2Transport):
    """REST backend transport for MetricsServiceV2.

    Service for configuring logs-based metrics.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    NOTE: This REST transport functionality is currently in a beta
    state (preview). We welcome your feedback via an issue in this
    library's source repository. Thank you!
    """

    def __init__(self, *, host: str='logging.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]]=None, quota_project_id: Optional[str]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO, always_use_jwt_access: Optional[bool]=False, url_scheme: str='https', interceptor: Optional[MetricsServiceV2RestInterceptor]=None, api_audience: Optional[str]=None) -> None:
        """Instantiate the transport.

       NOTE: This REST transport functionality is currently in a beta
       state (preview). We welcome your feedback via a GitHub issue in
       this library's repository. Thank you!

        Args:
            host (Optional[str]):
                 The hostname to connect to.
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        maybe_url_match = re.match('^(?P<scheme>http(?:s)?://)?(?P<host>.*)$', host)
        if maybe_url_match is None:
            raise ValueError(f'Unexpected hostname structure: {host}')
        url_match_items = maybe_url_match.groupdict()
        host = f'{url_scheme}://{host}' if not url_match_items['scheme'] else host
        super().__init__(host=host, credentials=credentials, client_info=client_info, always_use_jwt_access=always_use_jwt_access, api_audience=api_audience)
        self._session = AuthorizedSession(self._credentials, default_host=self.DEFAULT_HOST)
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or MetricsServiceV2RestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CreateLogMetric(MetricsServiceV2RestStub):

        def __hash__(self):
            return hash('CreateLogMetric')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging_metrics.CreateLogMetricRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging_metrics.LogMetric:
            """Call the create log metric method over HTTP.

            Args:
                request (~.logging_metrics.CreateLogMetricRequest):
                    The request object. The parameters to CreateLogMetric.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging_metrics.LogMetric:
                    Describes a logs-based metric. The
                value of the metric is the number of log
                entries that match a logs filter in a
                given time interval.

                Logs-based metrics can also be used to
                extract values from logs and create a
                distribution of the values. The
                distribution records the statistics of
                the extracted values along with an
                optional histogram of the values as
                specified by the bucket options.

            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v2/{parent=projects/*}/metrics', 'body': 'metric'}]
            request, metadata = self._interceptor.pre_create_log_metric(request, metadata)
            pb_request = logging_metrics.CreateLogMetricRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=False)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging_metrics.LogMetric()
            pb_resp = logging_metrics.LogMetric.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_log_metric(resp)
            return resp

    class _DeleteLogMetric(MetricsServiceV2RestStub):

        def __hash__(self):
            return hash('DeleteLogMetric')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging_metrics.DeleteLogMetricRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            """Call the delete log metric method over HTTP.

            Args:
                request (~.logging_metrics.DeleteLogMetricRequest):
                    The request object. The parameters to DeleteLogMetric.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """
            http_options: List[Dict[str, str]] = [{'method': 'delete', 'uri': '/v2/{metric_name=projects/*/metrics/*}'}]
            request, metadata = self._interceptor.pre_delete_log_metric(request, metadata)
            pb_request = logging_metrics.DeleteLogMetricRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _GetLogMetric(MetricsServiceV2RestStub):

        def __hash__(self):
            return hash('GetLogMetric')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging_metrics.GetLogMetricRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging_metrics.LogMetric:
            """Call the get log metric method over HTTP.

            Args:
                request (~.logging_metrics.GetLogMetricRequest):
                    The request object. The parameters to GetLogMetric.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging_metrics.LogMetric:
                    Describes a logs-based metric. The
                value of the metric is the number of log
                entries that match a logs filter in a
                given time interval.

                Logs-based metrics can also be used to
                extract values from logs and create a
                distribution of the values. The
                distribution records the statistics of
                the extracted values along with an
                optional histogram of the values as
                specified by the bucket options.

            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v2/{metric_name=projects/*/metrics/*}'}]
            request, metadata = self._interceptor.pre_get_log_metric(request, metadata)
            pb_request = logging_metrics.GetLogMetricRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging_metrics.LogMetric()
            pb_resp = logging_metrics.LogMetric.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_log_metric(resp)
            return resp

    class _ListLogMetrics(MetricsServiceV2RestStub):

        def __hash__(self):
            return hash('ListLogMetrics')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging_metrics.ListLogMetricsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging_metrics.ListLogMetricsResponse:
            """Call the list log metrics method over HTTP.

            Args:
                request (~.logging_metrics.ListLogMetricsRequest):
                    The request object. The parameters to ListLogMetrics.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging_metrics.ListLogMetricsResponse:
                    Result returned from ListLogMetrics.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v2/{parent=projects/*}/metrics'}]
            request, metadata = self._interceptor.pre_list_log_metrics(request, metadata)
            pb_request = logging_metrics.ListLogMetricsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging_metrics.ListLogMetricsResponse()
            pb_resp = logging_metrics.ListLogMetricsResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_log_metrics(resp)
            return resp

    class _UpdateLogMetric(MetricsServiceV2RestStub):

        def __hash__(self):
            return hash('UpdateLogMetric')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging_metrics.UpdateLogMetricRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging_metrics.LogMetric:
            """Call the update log metric method over HTTP.

            Args:
                request (~.logging_metrics.UpdateLogMetricRequest):
                    The request object. The parameters to UpdateLogMetric.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging_metrics.LogMetric:
                    Describes a logs-based metric. The
                value of the metric is the number of log
                entries that match a logs filter in a
                given time interval.

                Logs-based metrics can also be used to
                extract values from logs and create a
                distribution of the values. The
                distribution records the statistics of
                the extracted values along with an
                optional histogram of the values as
                specified by the bucket options.

            """
            http_options: List[Dict[str, str]] = [{'method': 'put', 'uri': '/v2/{metric_name=projects/*/metrics/*}', 'body': 'metric'}]
            request, metadata = self._interceptor.pre_update_log_metric(request, metadata)
            pb_request = logging_metrics.UpdateLogMetricRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=False)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging_metrics.LogMetric()
            pb_resp = logging_metrics.LogMetric.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_log_metric(resp)
            return resp

    @property
    def create_log_metric(self) -> Callable[[logging_metrics.CreateLogMetricRequest], logging_metrics.LogMetric]:
        return self._CreateLogMetric(self._session, self._host, self._interceptor)

    @property
    def delete_log_metric(self) -> Callable[[logging_metrics.DeleteLogMetricRequest], empty_pb2.Empty]:
        return self._DeleteLogMetric(self._session, self._host, self._interceptor)

    @property
    def get_log_metric(self) -> Callable[[logging_metrics.GetLogMetricRequest], logging_metrics.LogMetric]:
        return self._GetLogMetric(self._session, self._host, self._interceptor)

    @property
    def list_log_metrics(self) -> Callable[[logging_metrics.ListLogMetricsRequest], logging_metrics.ListLogMetricsResponse]:
        return self._ListLogMetrics(self._session, self._host, self._interceptor)

    @property
    def update_log_metric(self) -> Callable[[logging_metrics.UpdateLogMetricRequest], logging_metrics.LogMetric]:
        return self._UpdateLogMetric(self._session, self._host, self._interceptor)

    @property
    def kind(self) -> str:
        return 'rest'

    def close(self):
        self._session.close()