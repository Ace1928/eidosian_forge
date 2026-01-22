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
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts_lrs
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import TextToSpeechLongAudioSynthesizeTransport
class TextToSpeechLongAudioSynthesizeRestTransport(TextToSpeechLongAudioSynthesizeTransport):
    """REST backend transport for TextToSpeechLongAudioSynthesize.

    Service that implements Google Cloud Text-to-Speech API.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    NOTE: This REST transport functionality is currently in a beta
    state (preview). We welcome your feedback via an issue in this
    library's source repository. Thank you!
    """

    def __init__(self, *, host: str='texttospeech.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]]=None, quota_project_id: Optional[str]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO, always_use_jwt_access: Optional[bool]=False, url_scheme: str='https', interceptor: Optional[TextToSpeechLongAudioSynthesizeRestInterceptor]=None, api_audience: Optional[str]=None) -> None:
        """Instantiate the transport.

        NOTE: This REST transport functionality is currently in a beta
        state (preview). We welcome your feedback via a GitHub issue in
        this library's repository. Thank you!

         Args:
             host (Optional[str]):
                  The hostname to connect to (default: 'texttospeech.googleapis.com').
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
        self._operations_client: Optional[operations_v1.AbstractOperationsClient] = None
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or TextToSpeechLongAudioSynthesizeRestInterceptor()
        self._prep_wrapped_messages(client_info)

    @property
    def operations_client(self) -> operations_v1.AbstractOperationsClient:
        """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
        if self._operations_client is None:
            http_options: Dict[str, List[Dict[str, str]]] = {'google.longrunning.Operations.GetOperation': [{'method': 'get', 'uri': '/v1/{name=projects/*/locations/*/operations/*}'}], 'google.longrunning.Operations.ListOperations': [{'method': 'get', 'uri': '/v1/{name=projects/*/locations/*}/operations'}]}
            rest_transport = operations_v1.OperationsRestTransport(host=self._host, credentials=self._credentials, scopes=self._scopes, http_options=http_options, path_prefix='v1')
            self._operations_client = operations_v1.AbstractOperationsClient(transport=rest_transport)
        return self._operations_client

    class _SynthesizeLongAudio(TextToSpeechLongAudioSynthesizeRestStub):

        def __hash__(self):
            return hash('SynthesizeLongAudio')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_tts_lrs.SynthesizeLongAudioRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> operations_pb2.Operation:
            """Call the synthesize long audio method over HTTP.

            Args:
                request (~.cloud_tts_lrs.SynthesizeLongAudioRequest):
                    The request object. The top-level message sent by the client for the
                ``SynthesizeLongAudio`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.operations_pb2.Operation:
                    This resource represents a
                long-running operation that is the
                result of a network API call.

            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1/{parent=projects/*/locations/*}:synthesizeLongAudio', 'body': '*'}]
            request, metadata = self._interceptor.pre_synthesize_long_audio(request, metadata)
            pb_request = cloud_tts_lrs.SynthesizeLongAudioRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], use_integers_for_enums=False)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = operations_pb2.Operation()
            json_format.Parse(response.content, resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_synthesize_long_audio(resp)
            return resp

    @property
    def synthesize_long_audio(self) -> Callable[[cloud_tts_lrs.SynthesizeLongAudioRequest], operations_pb2.Operation]:
        return self._SynthesizeLongAudio(self._session, self._host, self._interceptor)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)

    class _GetOperation(TextToSpeechLongAudioSynthesizeRestStub):

        def __call__(self, request: operations_pb2.GetOperationRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> operations_pb2.Operation:
            """Call the get operation method over HTTP.

            Args:
                request (operations_pb2.GetOperationRequest):
                    The request object for GetOperation method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                operations_pb2.Operation: Response from GetOperation method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{name=projects/*/locations/*/operations/*}'}]
            request, metadata = self._interceptor.pre_get_operation(request, metadata)
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
            resp = operations_pb2.Operation()
            resp = json_format.Parse(response.content.decode('utf-8'), resp)
            resp = self._interceptor.post_get_operation(resp)
            return resp

    @property
    def list_operations(self):
        return self._ListOperations(self._session, self._host, self._interceptor)

    class _ListOperations(TextToSpeechLongAudioSynthesizeRestStub):

        def __call__(self, request: operations_pb2.ListOperationsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> operations_pb2.ListOperationsResponse:
            """Call the list operations method over HTTP.

            Args:
                request (operations_pb2.ListOperationsRequest):
                    The request object for ListOperations method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                operations_pb2.ListOperationsResponse: Response from ListOperations method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{name=projects/*/locations/*}/operations'}]
            request, metadata = self._interceptor.pre_list_operations(request, metadata)
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
            resp = operations_pb2.ListOperationsResponse()
            resp = json_format.Parse(response.content.decode('utf-8'), resp)
            resp = self._interceptor.post_list_operations(resp)
            return resp

    @property
    def kind(self) -> str:
        return 'rest'

    def close(self):
        self._session.close()