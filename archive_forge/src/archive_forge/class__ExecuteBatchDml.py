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
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import commit_response
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import result_set
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import spanner
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import transaction
from .base import SpannerTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class _ExecuteBatchDml(SpannerRestStub):

    def __hash__(self):
        return hash('ExecuteBatchDml')
    __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

    @classmethod
    def _get_unset_required_fields(cls, message_dict):
        return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

    def __call__(self, request: spanner.ExecuteBatchDmlRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> spanner.ExecuteBatchDmlResponse:
        """Call the execute batch dml method over HTTP.

            Args:
                request (~.spanner.ExecuteBatchDmlRequest):
                    The request object. The request for
                [ExecuteBatchDml][google.spanner.v1.Spanner.ExecuteBatchDml].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.spanner.ExecuteBatchDmlResponse:
                    The response for
                [ExecuteBatchDml][google.spanner.v1.Spanner.ExecuteBatchDml].
                Contains a list of
                [ResultSet][google.spanner.v1.ResultSet] messages, one
                for each DML statement that has successfully executed,
                in the same order as the statements in the request. If a
                statement fails, the status in the response body
                identifies the cause of the failure.

                To check for DML statements that failed, use the
                following approach:

                1. Check the status in the response message. The
                   [google.rpc.Code][google.rpc.Code] enum value ``OK``
                   indicates that all statements were executed
                   successfully.
                2. If the status was not ``OK``, check the number of
                   result sets in the response. If the response contains
                   ``N`` [ResultSet][google.spanner.v1.ResultSet]
                   messages, then statement ``N+1`` in the request
                   failed.

                Example 1:

                -  Request: 5 DML statements, all executed successfully.
                -  Response: 5 [ResultSet][google.spanner.v1.ResultSet]
                   messages, with the status ``OK``.

                Example 2:

                -  Request: 5 DML statements. The third statement has a
                   syntax error.
                -  Response: 2 [ResultSet][google.spanner.v1.ResultSet]
                   messages, and a syntax error (``INVALID_ARGUMENT``)
                   status. The number of
                   [ResultSet][google.spanner.v1.ResultSet] messages
                   indicates that the third statement failed, and the
                   fourth and fifth statements were not executed.

            """
        http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1/{session=projects/*/instances/*/databases/*/sessions/*}:executeBatchDml', 'body': '*'}]
        request, metadata = self._interceptor.pre_execute_batch_dml(request, metadata)
        pb_request = spanner.ExecuteBatchDmlRequest.pb(request)
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
        resp = spanner.ExecuteBatchDmlResponse()
        pb_resp = spanner.ExecuteBatchDmlResponse.pb(resp)
        json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_execute_batch_dml(resp)
        return resp