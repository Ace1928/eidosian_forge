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
class MetricsServiceV2RestInterceptor:
    """Interceptor for MetricsServiceV2.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the MetricsServiceV2RestTransport.

    .. code-block:: python
        class MyCustomMetricsServiceV2Interceptor(MetricsServiceV2RestInterceptor):
            def pre_create_log_metric(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_log_metric(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_log_metric(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_log_metric(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_log_metric(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_log_metrics(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_log_metrics(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_log_metric(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_log_metric(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = MetricsServiceV2RestTransport(interceptor=MyCustomMetricsServiceV2Interceptor())
        client = MetricsServiceV2Client(transport=transport)


    """

    def pre_create_log_metric(self, request: logging_metrics.CreateLogMetricRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_metrics.CreateLogMetricRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_log_metric

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetricsServiceV2 server.
        """
        return (request, metadata)

    def post_create_log_metric(self, response: logging_metrics.LogMetric) -> logging_metrics.LogMetric:
        """Post-rpc interceptor for create_log_metric

        Override in a subclass to manipulate the response
        after it is returned by the MetricsServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_delete_log_metric(self, request: logging_metrics.DeleteLogMetricRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_metrics.DeleteLogMetricRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_log_metric

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetricsServiceV2 server.
        """
        return (request, metadata)

    def pre_get_log_metric(self, request: logging_metrics.GetLogMetricRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_metrics.GetLogMetricRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_log_metric

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetricsServiceV2 server.
        """
        return (request, metadata)

    def post_get_log_metric(self, response: logging_metrics.LogMetric) -> logging_metrics.LogMetric:
        """Post-rpc interceptor for get_log_metric

        Override in a subclass to manipulate the response
        after it is returned by the MetricsServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_log_metrics(self, request: logging_metrics.ListLogMetricsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_metrics.ListLogMetricsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_log_metrics

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetricsServiceV2 server.
        """
        return (request, metadata)

    def post_list_log_metrics(self, response: logging_metrics.ListLogMetricsResponse) -> logging_metrics.ListLogMetricsResponse:
        """Post-rpc interceptor for list_log_metrics

        Override in a subclass to manipulate the response
        after it is returned by the MetricsServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_update_log_metric(self, request: logging_metrics.UpdateLogMetricRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_metrics.UpdateLogMetricRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_log_metric

        Override in a subclass to manipulate the request or metadata
        before they are sent to the MetricsServiceV2 server.
        """
        return (request, metadata)

    def post_update_log_metric(self, response: logging_metrics.LogMetric) -> logging_metrics.LogMetric:
        """Post-rpc interceptor for update_log_metric

        Override in a subclass to manipulate the response
        after it is returned by the MetricsServiceV2 server but before
        it is returned to user code.
        """
        return response