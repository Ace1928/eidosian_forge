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
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging
from .base import LoggingServiceV2Transport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class LoggingServiceV2RestInterceptor:
    """Interceptor for LoggingServiceV2.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the LoggingServiceV2RestTransport.

    .. code-block:: python
        class MyCustomLoggingServiceV2Interceptor(LoggingServiceV2RestInterceptor):
            def pre_delete_log(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_list_log_entries(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_log_entries(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_logs(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_logs(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_monitored_resource_descriptors(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_monitored_resource_descriptors(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_write_log_entries(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_write_log_entries(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = LoggingServiceV2RestTransport(interceptor=MyCustomLoggingServiceV2Interceptor())
        client = LoggingServiceV2Client(transport=transport)


    """

    def pre_delete_log(self, request: logging.DeleteLogRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging.DeleteLogRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_log

        Override in a subclass to manipulate the request or metadata
        before they are sent to the LoggingServiceV2 server.
        """
        return (request, metadata)

    def pre_list_log_entries(self, request: logging.ListLogEntriesRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging.ListLogEntriesRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_log_entries

        Override in a subclass to manipulate the request or metadata
        before they are sent to the LoggingServiceV2 server.
        """
        return (request, metadata)

    def post_list_log_entries(self, response: logging.ListLogEntriesResponse) -> logging.ListLogEntriesResponse:
        """Post-rpc interceptor for list_log_entries

        Override in a subclass to manipulate the response
        after it is returned by the LoggingServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_logs(self, request: logging.ListLogsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging.ListLogsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_logs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the LoggingServiceV2 server.
        """
        return (request, metadata)

    def post_list_logs(self, response: logging.ListLogsResponse) -> logging.ListLogsResponse:
        """Post-rpc interceptor for list_logs

        Override in a subclass to manipulate the response
        after it is returned by the LoggingServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_monitored_resource_descriptors(self, request: logging.ListMonitoredResourceDescriptorsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging.ListMonitoredResourceDescriptorsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_monitored_resource_descriptors

        Override in a subclass to manipulate the request or metadata
        before they are sent to the LoggingServiceV2 server.
        """
        return (request, metadata)

    def post_list_monitored_resource_descriptors(self, response: logging.ListMonitoredResourceDescriptorsResponse) -> logging.ListMonitoredResourceDescriptorsResponse:
        """Post-rpc interceptor for list_monitored_resource_descriptors

        Override in a subclass to manipulate the response
        after it is returned by the LoggingServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_write_log_entries(self, request: logging.WriteLogEntriesRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging.WriteLogEntriesRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for write_log_entries

        Override in a subclass to manipulate the request or metadata
        before they are sent to the LoggingServiceV2 server.
        """
        return (request, metadata)

    def post_write_log_entries(self, response: logging.WriteLogEntriesResponse) -> logging.WriteLogEntriesResponse:
        """Post-rpc interceptor for write_log_entries

        Override in a subclass to manipulate the response
        after it is returned by the LoggingServiceV2 server but before
        it is returned to user code.
        """
        return response