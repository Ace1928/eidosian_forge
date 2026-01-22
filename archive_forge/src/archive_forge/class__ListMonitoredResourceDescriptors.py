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
class _ListMonitoredResourceDescriptors(LoggingServiceV2RestStub):

    def __hash__(self):
        return hash('ListMonitoredResourceDescriptors')

    def __call__(self, request: logging.ListMonitoredResourceDescriptorsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging.ListMonitoredResourceDescriptorsResponse:
        """Call the list monitored resource
        descriptors method over HTTP.

            Args:
                request (~.logging.ListMonitoredResourceDescriptorsRequest):
                    The request object. The parameters to
                ListMonitoredResourceDescriptors
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging.ListMonitoredResourceDescriptorsResponse:
                    Result returned from
                ListMonitoredResourceDescriptors.

            """
        http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v2/monitoredResourceDescriptors'}]
        request, metadata = self._interceptor.pre_list_monitored_resource_descriptors(request, metadata)
        pb_request = logging.ListMonitoredResourceDescriptorsRequest.pb(request)
        transcoded_request = path_template.transcode(http_options, pb_request)
        uri = transcoded_request['uri']
        method = transcoded_request['method']
        query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
        headers = dict(metadata)
        headers['Content-Type'] = 'application/json'
        response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)
        resp = logging.ListMonitoredResourceDescriptorsResponse()
        pb_resp = logging.ListMonitoredResourceDescriptorsResponse.pb(resp)
        json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_list_monitored_resource_descriptors(resp)
        return resp