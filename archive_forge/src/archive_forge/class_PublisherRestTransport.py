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
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from google.pubsub_v1.types import pubsub
from .base import PublisherTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class PublisherRestTransport(PublisherTransport):
    """REST backend transport for Publisher.

    The service that an application uses to manipulate topics,
    and to send messages to a topic.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    """

    def __init__(self, *, host: str='pubsub.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]]=None, quota_project_id: Optional[str]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO, always_use_jwt_access: Optional[bool]=False, url_scheme: str='https', interceptor: Optional[PublisherRestInterceptor]=None, api_audience: Optional[str]=None) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to.
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
        self._interceptor = interceptor or PublisherRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CreateTopic(PublisherRestStub):

        def __hash__(self):
            return hash('CreateTopic')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.Topic, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.Topic:
            """Call the create topic method over HTTP.

            Args:
                request (~.pubsub.Topic):
                    The request object. A topic resource.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.Topic:
                    A topic resource.
            """
            http_options: List[Dict[str, str]] = [{'method': 'put', 'uri': '/v1/{name=projects/*/topics/*}', 'body': '*'}]
            request, metadata = self._interceptor.pre_create_topic(request, metadata)
            pb_request = pubsub.Topic.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=True)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.Topic()
            pb_resp = pubsub.Topic.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_topic(resp)
            return resp

    class _DeleteTopic(PublisherRestStub):

        def __hash__(self):
            return hash('DeleteTopic')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.DeleteTopicRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            """Call the delete topic method over HTTP.

            Args:
                request (~.pubsub.DeleteTopicRequest):
                    The request object. Request for the ``DeleteTopic`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """
            http_options: List[Dict[str, str]] = [{'method': 'delete', 'uri': '/v1/{topic=projects/*/topics/*}'}]
            request, metadata = self._interceptor.pre_delete_topic(request, metadata)
            pb_request = pubsub.DeleteTopicRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DetachSubscription(PublisherRestStub):

        def __hash__(self):
            return hash('DetachSubscription')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.DetachSubscriptionRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.DetachSubscriptionResponse:
            """Call the detach subscription method over HTTP.

            Args:
                request (~.pubsub.DetachSubscriptionRequest):
                    The request object. Request for the DetachSubscription
                method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.DetachSubscriptionResponse:
                    Response for the DetachSubscription
                method. Reserved for future use.

            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1/{subscription=projects/*/subscriptions/*}:detach'}]
            request, metadata = self._interceptor.pre_detach_subscription(request, metadata)
            pb_request = pubsub.DetachSubscriptionRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.DetachSubscriptionResponse()
            pb_resp = pubsub.DetachSubscriptionResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_detach_subscription(resp)
            return resp

    class _GetTopic(PublisherRestStub):

        def __hash__(self):
            return hash('GetTopic')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.GetTopicRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.Topic:
            """Call the get topic method over HTTP.

            Args:
                request (~.pubsub.GetTopicRequest):
                    The request object. Request for the GetTopic method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.Topic:
                    A topic resource.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{topic=projects/*/topics/*}'}]
            request, metadata = self._interceptor.pre_get_topic(request, metadata)
            pb_request = pubsub.GetTopicRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.Topic()
            pb_resp = pubsub.Topic.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_topic(resp)
            return resp

    class _ListTopics(PublisherRestStub):

        def __hash__(self):
            return hash('ListTopics')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.ListTopicsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.ListTopicsResponse:
            """Call the list topics method over HTTP.

            Args:
                request (~.pubsub.ListTopicsRequest):
                    The request object. Request for the ``ListTopics`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.ListTopicsResponse:
                    Response for the ``ListTopics`` method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{project=projects/*}/topics'}]
            request, metadata = self._interceptor.pre_list_topics(request, metadata)
            pb_request = pubsub.ListTopicsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.ListTopicsResponse()
            pb_resp = pubsub.ListTopicsResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_topics(resp)
            return resp

    class _ListTopicSnapshots(PublisherRestStub):

        def __hash__(self):
            return hash('ListTopicSnapshots')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.ListTopicSnapshotsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.ListTopicSnapshotsResponse:
            """Call the list topic snapshots method over HTTP.

            Args:
                request (~.pubsub.ListTopicSnapshotsRequest):
                    The request object. Request for the ``ListTopicSnapshots`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.ListTopicSnapshotsResponse:
                    Response for the ``ListTopicSnapshots`` method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{topic=projects/*/topics/*}/snapshots'}]
            request, metadata = self._interceptor.pre_list_topic_snapshots(request, metadata)
            pb_request = pubsub.ListTopicSnapshotsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.ListTopicSnapshotsResponse()
            pb_resp = pubsub.ListTopicSnapshotsResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_topic_snapshots(resp)
            return resp

    class _ListTopicSubscriptions(PublisherRestStub):

        def __hash__(self):
            return hash('ListTopicSubscriptions')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.ListTopicSubscriptionsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.ListTopicSubscriptionsResponse:
            """Call the list topic subscriptions method over HTTP.

            Args:
                request (~.pubsub.ListTopicSubscriptionsRequest):
                    The request object. Request for the ``ListTopicSubscriptions`` method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.ListTopicSubscriptionsResponse:
                    Response for the ``ListTopicSubscriptions`` method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{topic=projects/*/topics/*}/subscriptions'}]
            request, metadata = self._interceptor.pre_list_topic_subscriptions(request, metadata)
            pb_request = pubsub.ListTopicSubscriptionsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.ListTopicSubscriptionsResponse()
            pb_resp = pubsub.ListTopicSubscriptionsResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_topic_subscriptions(resp)
            return resp

    class _Publish(PublisherRestStub):

        def __hash__(self):
            return hash('Publish')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.PublishRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.PublishResponse:
            """Call the publish method over HTTP.

            Args:
                request (~.pubsub.PublishRequest):
                    The request object. Request for the Publish method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.PublishResponse:
                    Response for the ``Publish`` method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1/{topic=projects/*/topics/*}:publish', 'body': '*'}]
            request, metadata = self._interceptor.pre_publish(request, metadata)
            pb_request = pubsub.PublishRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=True)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.PublishResponse()
            pb_resp = pubsub.PublishResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_publish(resp)
            return resp

    class _UpdateTopic(PublisherRestStub):

        def __hash__(self):
            return hash('UpdateTopic')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: pubsub.UpdateTopicRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> pubsub.Topic:
            """Call the update topic method over HTTP.

            Args:
                request (~.pubsub.UpdateTopicRequest):
                    The request object. Request for the UpdateTopic method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.pubsub.Topic:
                    A topic resource.
            """
            http_options: List[Dict[str, str]] = [{'method': 'patch', 'uri': '/v1/{topic.name=projects/*/topics/*}', 'body': '*'}]
            request, metadata = self._interceptor.pre_update_topic(request, metadata)
            pb_request = pubsub.UpdateTopicRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=True)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=True))
            query_params.update(self._get_unset_required_fields(query_params))
            query_params['$alt'] = 'json;enum-encoding=int'
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = pubsub.Topic()
            pb_resp = pubsub.Topic.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_topic(resp)
            return resp

    @property
    def create_topic(self) -> Callable[[pubsub.Topic], pubsub.Topic]:
        return self._CreateTopic(self._session, self._host, self._interceptor)

    @property
    def delete_topic(self) -> Callable[[pubsub.DeleteTopicRequest], empty_pb2.Empty]:
        return self._DeleteTopic(self._session, self._host, self._interceptor)

    @property
    def detach_subscription(self) -> Callable[[pubsub.DetachSubscriptionRequest], pubsub.DetachSubscriptionResponse]:
        return self._DetachSubscription(self._session, self._host, self._interceptor)

    @property
    def get_topic(self) -> Callable[[pubsub.GetTopicRequest], pubsub.Topic]:
        return self._GetTopic(self._session, self._host, self._interceptor)

    @property
    def list_topics(self) -> Callable[[pubsub.ListTopicsRequest], pubsub.ListTopicsResponse]:
        return self._ListTopics(self._session, self._host, self._interceptor)

    @property
    def list_topic_snapshots(self) -> Callable[[pubsub.ListTopicSnapshotsRequest], pubsub.ListTopicSnapshotsResponse]:
        return self._ListTopicSnapshots(self._session, self._host, self._interceptor)

    @property
    def list_topic_subscriptions(self) -> Callable[[pubsub.ListTopicSubscriptionsRequest], pubsub.ListTopicSubscriptionsResponse]:
        return self._ListTopicSubscriptions(self._session, self._host, self._interceptor)

    @property
    def publish(self) -> Callable[[pubsub.PublishRequest], pubsub.PublishResponse]:
        return self._Publish(self._session, self._host, self._interceptor)

    @property
    def update_topic(self) -> Callable[[pubsub.UpdateTopicRequest], pubsub.Topic]:
        return self._UpdateTopic(self._session, self._host, self._interceptor)

    @property
    def get_iam_policy(self):
        return self._GetIamPolicy(self._session, self._host, self._interceptor)

    class _GetIamPolicy(PublisherRestStub):

        def __call__(self, request: iam_policy_pb2.GetIamPolicyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> policy_pb2.Policy:
            """Call the get iam policy method over HTTP.

            Args:
                request (iam_policy_pb2.GetIamPolicyRequest):
                    The request object for GetIamPolicy method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                policy_pb2.Policy: Response from GetIamPolicy method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v1/{resource=projects/*/topics/*}:getIamPolicy'}, {'method': 'get', 'uri': '/v1/{resource=projects/*/subscriptions/*}:getIamPolicy'}, {'method': 'get', 'uri': '/v1/{resource=projects/*/snapshots/*}:getIamPolicy'}, {'method': 'get', 'uri': '/v1/{resource=projects/*/schemas/*}:getIamPolicy'}]
            request, metadata = self._interceptor.pre_get_iam_policy(request, metadata)
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
            resp = policy_pb2.Policy()
            resp = json_format.Parse(response.content.decode('utf-8'), resp)
            resp = self._interceptor.post_get_iam_policy(resp)
            return resp

    @property
    def set_iam_policy(self):
        return self._SetIamPolicy(self._session, self._host, self._interceptor)

    class _SetIamPolicy(PublisherRestStub):

        def __call__(self, request: iam_policy_pb2.SetIamPolicyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> policy_pb2.Policy:
            """Call the set iam policy method over HTTP.

            Args:
                request (iam_policy_pb2.SetIamPolicyRequest):
                    The request object for SetIamPolicy method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                policy_pb2.Policy: Response from SetIamPolicy method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1/{resource=projects/*/topics/*}:setIamPolicy', 'body': '*'}, {'method': 'post', 'uri': '/v1/{resource=projects/*/subscriptions/*}:setIamPolicy', 'body': '*'}, {'method': 'post', 'uri': '/v1/{resource=projects/*/snapshots/*}:setIamPolicy', 'body': '*'}, {'method': 'post', 'uri': '/v1/{resource=projects/*/schemas/*}:setIamPolicy', 'body': '*'}]
            request, metadata = self._interceptor.pre_set_iam_policy(request, metadata)
            request_kwargs = json_format.MessageToDict(request)
            transcoded_request = path_template.transcode(http_options, **request_kwargs)
            body = json.dumps(transcoded_request['body'])
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json.dumps(transcoded_request['query_params']))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = policy_pb2.Policy()
            resp = json_format.Parse(response.content.decode('utf-8'), resp)
            resp = self._interceptor.post_set_iam_policy(resp)
            return resp

    @property
    def test_iam_permissions(self):
        return self._TestIamPermissions(self._session, self._host, self._interceptor)

    class _TestIamPermissions(PublisherRestStub):

        def __call__(self, request: iam_policy_pb2.TestIamPermissionsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> iam_policy_pb2.TestIamPermissionsResponse:
            """Call the test iam permissions method over HTTP.

            Args:
                request (iam_policy_pb2.TestIamPermissionsRequest):
                    The request object for TestIamPermissions method.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                iam_policy_pb2.TestIamPermissionsResponse: Response from TestIamPermissions method.
            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v1/{resource=projects/*/subscriptions/*}:testIamPermissions', 'body': '*'}, {'method': 'post', 'uri': '/v1/{resource=projects/*/topics/*}:testIamPermissions', 'body': '*'}, {'method': 'post', 'uri': '/v1/{resource=projects/*/snapshots/*}:testIamPermissions', 'body': '*'}, {'method': 'post', 'uri': '/v1/{resource=projects/*/schemas/*}:testIamPermissions', 'body': '*'}]
            request, metadata = self._interceptor.pre_test_iam_permissions(request, metadata)
            request_kwargs = json_format.MessageToDict(request)
            transcoded_request = path_template.transcode(http_options, **request_kwargs)
            body = json.dumps(transcoded_request['body'])
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json.dumps(transcoded_request['query_params']))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = iam_policy_pb2.TestIamPermissionsResponse()
            resp = json_format.Parse(response.content.decode('utf-8'), resp)
            resp = self._interceptor.post_test_iam_permissions(resp)
            return resp

    @property
    def kind(self) -> str:
        return 'rest'

    def close(self):
        self._session.close()