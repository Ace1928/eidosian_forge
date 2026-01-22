import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import (
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.speech_v2.types import cloud_speech
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import SpeechTransport
class _GetLocation(SpeechRestStub):

    def __call__(self, request: locations_pb2.GetLocationRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> locations_pb2.Location:
        """Call the get location method over HTTP.

            Args:
                request (locations_pb2.GetLocationRequest):
                    The request object for GetLocation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                locations_pb2.Location: Response from GetLocation method.
            """
        http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v2/{name=projects/*/locations/*}'}]
        request, metadata = self._interceptor.pre_get_location(request, metadata)
        request_kwargs = json_format.MessageToDict(request)
        transcoded_request = path_template.transcode(http_options, **request_kwargs)
        uri = transcoded_request['uri']
        method = transcoded_request['method']
        query_params = json.loads(json.dumps(transcoded_request['query_params']))
        headers = dict(metadata)
        headers['Content-Type'] = 'application/json'
        response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params))
        if response.status_code >= 400:
            raise core_exceptions.from_http_response(response)
        resp = locations_pb2.Location()
        resp = json_format.Parse(response.content.decode('utf-8'), resp)
        resp = self._interceptor.post_get_location(resp)
        return resp