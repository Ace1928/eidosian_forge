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
from google.api_core import operations_v1
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging_config
from google.longrunning import operations_pb2  # type: ignore
from .base import ConfigServiceV2Transport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class _UpdateExclusion(ConfigServiceV2RestStub):

    def __hash__(self):
        return hash('UpdateExclusion')
    __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {'updateMask': {}}

    @classmethod
    def _get_unset_required_fields(cls, message_dict):
        return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

    def __call__(self, request: logging_config.UpdateExclusionRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging_config.LogExclusion:
        """Call the update exclusion method over HTTP.

            Args:
                request (~.logging_config.UpdateExclusionRequest):
                    The request object. The parameters to ``UpdateExclusion``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging_config.LogExclusion:
                    Specifies a set of log entries that are filtered out by
                a sink. If your Google Cloud resource receives a large
                volume of log entries, you can use exclusions to reduce
                your chargeable logs. Note that exclusions on
                organization-level and folder-level sinks don't apply to
                child resources. Note also that you cannot modify the
                \\_Required sink or exclude logs from it.

            """
        http_options: List[Dict[str, str]] = [{'method': 'patch', 'uri': '/v2/{name=*/*/exclusions/*}', 'body': 'exclusion'}, {'method': 'patch', 'uri': '/v2/{name=projects/*/exclusions/*}', 'body': 'exclusion'}, {'method': 'patch', 'uri': '/v2/{name=organizations/*/exclusions/*}', 'body': 'exclusion'}, {'method': 'patch', 'uri': '/v2/{name=folders/*/exclusions/*}', 'body': 'exclusion'}, {'method': 'patch', 'uri': '/v2/{name=billingAccounts/*/exclusions/*}', 'body': 'exclusion'}]
        request, metadata = self._interceptor.pre_update_exclusion(request, metadata)
        pb_request = logging_config.UpdateExclusionRequest.pb(request)
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
        resp = logging_config.LogExclusion()
        pb_resp = logging_config.LogExclusion.pb(resp)
        json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
        resp = self._interceptor.post_update_exclusion(resp)
        return resp