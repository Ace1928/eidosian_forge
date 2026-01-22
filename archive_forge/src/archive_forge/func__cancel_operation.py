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
def _cancel_operation(self, request: operations_pb2.CancelOperationRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, compression: Optional[grpc.Compression]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> empty_pb2.Empty:
    """Call the cancel operation method over HTTP.

        Args:
            request (~.operations_pb2.CancelOperationRequest):
                The request object. The request message for
                [Operations.CancelOperation][google.api_core.operations_v1.Operations.CancelOperation].

            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
    http_options = [{'method': 'post', 'uri': '/{}/{{name=**/operations/*}}:cancel'.format(self._path_prefix), 'body': '*'}]
    if 'google.longrunning.Operations.CancelOperation' in self._http_options:
        http_options = self._http_options['google.longrunning.Operations.CancelOperation']
    request_kwargs = json_format.MessageToDict(request, preserving_proto_field_name=True, including_default_value_fields=True)
    transcoded_request = path_template.transcode(http_options, **request_kwargs)
    body_request = operations_pb2.CancelOperationRequest()
    json_format.ParseDict(transcoded_request['body'], body_request)
    body = json_format.MessageToDict(body_request, including_default_value_fields=False, preserving_proto_field_name=False, use_integers_for_enums=False)
    uri = transcoded_request['uri']
    method = transcoded_request['method']
    query_params_request = operations_pb2.CancelOperationRequest()
    json_format.ParseDict(transcoded_request['query_params'], query_params_request)
    query_params = json_format.MessageToDict(query_params_request, including_default_value_fields=False, preserving_proto_field_name=False, use_integers_for_enums=False)
    headers = dict(metadata)
    headers['Content-Type'] = 'application/json'
    response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params), data=body)
    if response.status_code >= 400:
        raise core_exceptions.from_http_response(response)
    return empty_pb2.Empty()