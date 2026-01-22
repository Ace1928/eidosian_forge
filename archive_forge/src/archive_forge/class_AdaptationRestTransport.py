import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import gapic_v1, path_template, rest_helpers, rest_streaming
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.cloud.speech_v1p1beta1.types import cloud_speech_adaptation, resource
from .base import AdaptationTransport
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class AdaptationRestTransport(AdaptationTransport):
    """REST backend transport for Adaptation.

    Service that implements Google Cloud Speech Adaptation API.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    """

    def __init__(self, *, host: str='speech.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]]=None, quota_project_id: Optional[str]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO, always_use_jwt_access: Optional[bool]=False, url_scheme: str='https', interceptor: Optional[AdaptationRestInterceptor]=None, api_audience: Optional[str]=None) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'speech.googleapis.com').
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
        self._interceptor = interceptor or AdaptationRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CreateCustomClass(AdaptationRestStub):

        def __hash__(self):
            return hash('CreateCustomClass')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.CreateCustomClassRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> resource.CustomClass:
            """Call the create custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.CreateCustomClassRequest):
                    The request object. Message sent by the client for the ``CreateCustomClass``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.resource.CustomClass:
                    A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1p1beta1/{parent=projects/*/locations/*}/customClasses', 'body': '*'}]
            request, metadata = self._interceptor.pre_create_custom_class(request, metadata)
            pb_request = cloud_speech_adaptation.CreateCustomClassRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], use_integers_for_enums=True)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = resource.CustomClass()
            pb_resp = resource.CustomClass.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_custom_class(resp)
            return resp

    class _CreatePhraseSet(AdaptationRestStub):

        def __hash__(self):
            return hash('CreatePhraseSet')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.CreatePhraseSetRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> resource.PhraseSet:
            """Call the create phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.CreatePhraseSetRequest):
                    The request object. Message sent by the client for the ``CreatePhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.resource.PhraseSet:
                    Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1p1beta1/{parent=projects/*/locations/*}/phraseSets', 'body': '*'}]
            request, metadata = self._interceptor.pre_create_phrase_set(request, metadata)
            pb_request = cloud_speech_adaptation.CreatePhraseSetRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], use_integers_for_enums=True)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = resource.PhraseSet()
            pb_resp = resource.PhraseSet.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_phrase_set(resp)
            return resp

    class _DeleteCustomClass(AdaptationRestStub):

        def __hash__(self):
            return hash('DeleteCustomClass')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.DeleteCustomClassRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            """Call the delete custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.DeleteCustomClassRequest):
                    The request object. Message sent by the client for the ``DeleteCustomClass``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """
            http_options: List[Dict[str, str]] = [{'method': 'delete', 'uri': '/v1p1beta1/{name=projects/*/locations/*/customClasses/*}'}]
            request, metadata = self._interceptor.pre_delete_custom_class(request, metadata)
            pb_request = cloud_speech_adaptation.DeleteCustomClassRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeletePhraseSet(AdaptationRestStub):

        def __hash__(self):
            return hash('DeletePhraseSet')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.DeletePhraseSetRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            """Call the delete phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.DeletePhraseSetRequest):
                    The request object. Message sent by the client for the ``DeletePhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """
            http_options: List[Dict[str, str]] = [{'method': 'delete', 'uri': '/v1p1beta1/{name=projects/*/locations/*/phraseSets/*}'}]
            request, metadata = self._interceptor.pre_delete_phrase_set(request, metadata)
            pb_request = cloud_speech_adaptation.DeletePhraseSetRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _GetCustomClass(AdaptationRestStub):

        def __hash__(self):
            return hash('GetCustomClass')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.GetCustomClassRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> resource.CustomClass:
            """Call the get custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.GetCustomClassRequest):
                    The request object. Message sent by the client for the ``GetCustomClass``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.resource.CustomClass:
                    A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1p1beta1/{name=projects/*/locations/*/customClasses/*}'}]
            request, metadata = self._interceptor.pre_get_custom_class(request, metadata)
            pb_request = cloud_speech_adaptation.GetCustomClassRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = resource.CustomClass()
            pb_resp = resource.CustomClass.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_custom_class(resp)
            return resp

    class _GetPhraseSet(AdaptationRestStub):

        def __hash__(self):
            return hash('GetPhraseSet')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.GetPhraseSetRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> resource.PhraseSet:
            """Call the get phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.GetPhraseSetRequest):
                    The request object. Message sent by the client for the ``GetPhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.resource.PhraseSet:
                    Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1p1beta1/{name=projects/*/locations/*/phraseSets/*}'}]
            request, metadata = self._interceptor.pre_get_phrase_set(request, metadata)
            pb_request = cloud_speech_adaptation.GetPhraseSetRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = resource.PhraseSet()
            pb_resp = resource.PhraseSet.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_phrase_set(resp)
            return resp

    class _ListCustomClasses(AdaptationRestStub):

        def __hash__(self):
            return hash('ListCustomClasses')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.ListCustomClassesRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> cloud_speech_adaptation.ListCustomClassesResponse:
            """Call the list custom classes method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.ListCustomClassesRequest):
                    The request object. Message sent by the client for the ``ListCustomClasses``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.cloud_speech_adaptation.ListCustomClassesResponse:
                    Message returned to the client by the
                ``ListCustomClasses`` method.

            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1p1beta1/{parent=projects/*/locations/*}/customClasses'}]
            request, metadata = self._interceptor.pre_list_custom_classes(request, metadata)
            pb_request = cloud_speech_adaptation.ListCustomClassesRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = cloud_speech_adaptation.ListCustomClassesResponse()
            pb_resp = cloud_speech_adaptation.ListCustomClassesResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_custom_classes(resp)
            return resp

    class _ListPhraseSet(AdaptationRestStub):

        def __hash__(self):
            return hash('ListPhraseSet')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.ListPhraseSetRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> cloud_speech_adaptation.ListPhraseSetResponse:
            """Call the list phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.ListPhraseSetRequest):
                    The request object. Message sent by the client for the ``ListPhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.cloud_speech_adaptation.ListPhraseSetResponse:
                    Message returned to the client by the ``ListPhraseSet``
                method.

            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1p1beta1/{parent=projects/*/locations/*}/phraseSets'}]
            request, metadata = self._interceptor.pre_list_phrase_set(request, metadata)
            pb_request = cloud_speech_adaptation.ListPhraseSetRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = cloud_speech_adaptation.ListPhraseSetResponse()
            pb_resp = cloud_speech_adaptation.ListPhraseSetResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_phrase_set(resp)
            return resp

    class _UpdateCustomClass(AdaptationRestStub):

        def __hash__(self):
            return hash('UpdateCustomClass')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.UpdateCustomClassRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> resource.CustomClass:
            """Call the update custom class method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.UpdateCustomClassRequest):
                    The request object. Message sent by the client for the ``UpdateCustomClass``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.resource.CustomClass:
                    A set of words or phrases that
                represents a common concept likely to
                appear in your audio, for example a list
                of passenger ship names. CustomClass
                items can be substituted into
                placeholders that you set in PhraseSet
                phrases.

            """
            http_options: List[Dict[str, str]] = [{'method': 'patch', 'uri': '/v1p1beta1/{custom_class.name=projects/*/locations/*/customClasses/*}', 'body': 'custom_class'}]
            request, metadata = self._interceptor.pre_update_custom_class(request, metadata)
            pb_request = cloud_speech_adaptation.UpdateCustomClassRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], use_integers_for_enums=True)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = resource.CustomClass()
            pb_resp = resource.CustomClass.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_custom_class(resp)
            return resp

    class _UpdatePhraseSet(AdaptationRestStub):

        def __hash__(self):
            return hash('UpdatePhraseSet')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: cloud_speech_adaptation.UpdatePhraseSetRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> resource.PhraseSet:
            """Call the update phrase set method over HTTP.

            Args:
                request (~.cloud_speech_adaptation.UpdatePhraseSetRequest):
                    The request object. Message sent by the client for the ``UpdatePhraseSet``
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.resource.PhraseSet:
                    Provides "hints" to the speech
                recognizer to favor specific words and
                phrases in the results.

            """
            http_options: List[Dict[str, str]] = [{'method': 'patch', 'uri': '/v1p1beta1/{phrase_set.name=projects/*/locations/*/phraseSets/*}', 'body': 'phrase_set'}]
            request, metadata = self._interceptor.pre_update_phrase_set(request, metadata)
            pb_request = cloud_speech_adaptation.UpdatePhraseSetRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], use_integers_for_enums=True)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = resource.PhraseSet()
            pb_resp = resource.PhraseSet.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_phrase_set(resp)
            return resp

    @property
    def create_custom_class(self) -> Callable[[cloud_speech_adaptation.CreateCustomClassRequest], resource.CustomClass]:
        return self._CreateCustomClass(self._session, self._host, self._interceptor)

    @property
    def create_phrase_set(self) -> Callable[[cloud_speech_adaptation.CreatePhraseSetRequest], resource.PhraseSet]:
        return self._CreatePhraseSet(self._session, self._host, self._interceptor)

    @property
    def delete_custom_class(self) -> Callable[[cloud_speech_adaptation.DeleteCustomClassRequest], empty_pb2.Empty]:
        return self._DeleteCustomClass(self._session, self._host, self._interceptor)

    @property
    def delete_phrase_set(self) -> Callable[[cloud_speech_adaptation.DeletePhraseSetRequest], empty_pb2.Empty]:
        return self._DeletePhraseSet(self._session, self._host, self._interceptor)

    @property
    def get_custom_class(self) -> Callable[[cloud_speech_adaptation.GetCustomClassRequest], resource.CustomClass]:
        return self._GetCustomClass(self._session, self._host, self._interceptor)

    @property
    def get_phrase_set(self) -> Callable[[cloud_speech_adaptation.GetPhraseSetRequest], resource.PhraseSet]:
        return self._GetPhraseSet(self._session, self._host, self._interceptor)

    @property
    def list_custom_classes(self) -> Callable[[cloud_speech_adaptation.ListCustomClassesRequest], cloud_speech_adaptation.ListCustomClassesResponse]:
        return self._ListCustomClasses(self._session, self._host, self._interceptor)

    @property
    def list_phrase_set(self) -> Callable[[cloud_speech_adaptation.ListPhraseSetRequest], cloud_speech_adaptation.ListPhraseSetResponse]:
        return self._ListPhraseSet(self._session, self._host, self._interceptor)

    @property
    def update_custom_class(self) -> Callable[[cloud_speech_adaptation.UpdateCustomClassRequest], resource.CustomClass]:
        return self._UpdateCustomClass(self._session, self._host, self._interceptor)

    @property
    def update_phrase_set(self) -> Callable[[cloud_speech_adaptation.UpdatePhraseSetRequest], resource.PhraseSet]:
        return self._UpdatePhraseSet(self._session, self._host, self._interceptor)

    @property
    def get_operation(self):
        return self._GetOperation(self._session, self._host, self._interceptor)

    class _GetOperation(AdaptationRestStub):

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
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1p1beta1/operations/{name=**}'}]
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

    class _ListOperations(AdaptationRestStub):

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
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1p1beta1/operations'}]
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