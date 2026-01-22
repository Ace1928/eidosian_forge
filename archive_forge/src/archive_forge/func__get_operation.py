import re
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from requests import __version__ as requests_version
from google.api_core import exceptions as core_exceptions  # type: ignore
from google.api_core import gapic_v1  # type: ignore
from google.api_core import path_template  # type: ignore
from google.api_core import rest_helpers  # type: ignore
from google.api_core import retry as retries  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from cloudsdk.google.protobuf import json_format  # type: ignore
import grpc
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO, OperationsTransport
def _get_operation(self, request: operations_pb2.GetOperationRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, compression: Optional[grpc.Compression]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> operations_pb2.Operation:
    """Call the get operation method over HTTP.

        Args:
            request (~.operations_pb2.GetOperationRequest):
                The request object. The request message for
                [Operations.GetOperation][google.api_core.operations_v1.Operations.GetOperation].

            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            ~.operations_pb2.Operation:
                This resource represents a long-
                running operation that is the result of a
                network API call.

        """
    http_options = [{'method': 'get', 'uri': '/{}/{{name=**/operations/*}}'.format(self._path_prefix)}]
    if 'google.longrunning.Operations.GetOperation' in self._http_options:
        http_options = self._http_options['google.longrunning.Operations.GetOperation']
    request_kwargs = json_format.MessageToDict(request, preserving_proto_field_name=True, including_default_value_fields=True)
    transcoded_request = path_template.transcode(http_options, **request_kwargs)
    uri = transcoded_request['uri']
    method = transcoded_request['method']
    query_params_request = operations_pb2.GetOperationRequest()
    json_format.ParseDict(transcoded_request['query_params'], query_params_request)
    query_params = json_format.MessageToDict(query_params_request, including_default_value_fields=False, preserving_proto_field_name=False, use_integers_for_enums=False)
    headers = dict(metadata)
    headers['Content-Type'] = 'application/json'
    response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params))
    if response.status_code >= 400:
        raise core_exceptions.from_http_response(response)
    api_response = operations_pb2.Operation()
    json_format.Parse(response.content, api_response, ignore_unknown_fields=False)
    return api_response