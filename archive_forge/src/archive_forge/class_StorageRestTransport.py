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
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.storage_v2.types import storage
from .base import StorageTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class StorageRestTransport(StorageTransport):
    """REST backend transport for Storage.

    API Overview and Naming Syntax
    ------------------------------

    The Cloud Storage gRPC API allows applications to read and write
    data through the abstractions of buckets and objects. For a
    description of these abstractions please see
    https://cloud.google.com/storage/docs.

    Resources are named as follows:

    -  Projects are referred to as they are defined by the Resource
       Manager API, using strings like ``projects/123456`` or
       ``projects/my-string-id``.

    -  Buckets are named using string names of the form:
       ``projects/{project}/buckets/{bucket}`` For globally unique
       buckets, ``_`` may be substituted for the project.

    -  Objects are uniquely identified by their name along with the name
       of the bucket they belong to, as separate strings in this API.
       For example:

       ReadObjectRequest { bucket: 'projects/_/buckets/my-bucket'
       object: 'my-object' } Note that object names can contain ``/``
       characters, which are treated as any other character (no special
       directory semantics).

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    NOTE: This REST transport functionality is currently in a beta
    state (preview). We welcome your feedback via an issue in this
    library's source repository. Thank you!
    """

    def __init__(self, *, host: str='storage.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]]=None, quota_project_id: Optional[str]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO, always_use_jwt_access: Optional[bool]=False, url_scheme: str='https', interceptor: Optional[StorageRestInterceptor]=None, api_audience: Optional[str]=None) -> None:
        """Instantiate the transport.

       NOTE: This REST transport functionality is currently in a beta
       state (preview). We welcome your feedback via a GitHub issue in
       this library's repository. Thank you!

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
        self._interceptor = interceptor or StorageRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _BidiWriteObject(StorageRestStub):

        def __hash__(self):
            return hash('BidiWriteObject')

        def __call__(self, request: storage.BidiWriteObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> rest_streaming.ResponseIterator:
            raise NotImplementedError('Method BidiWriteObject is not available over REST transport')

    class _CancelResumableWrite(StorageRestStub):

        def __hash__(self):
            return hash('CancelResumableWrite')

        def __call__(self, request: storage.CancelResumableWriteRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.CancelResumableWriteResponse:
            raise NotImplementedError('Method CancelResumableWrite is not available over REST transport')

    class _ComposeObject(StorageRestStub):

        def __hash__(self):
            return hash('ComposeObject')

        def __call__(self, request: storage.ComposeObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Object:
            raise NotImplementedError('Method ComposeObject is not available over REST transport')

    class _CreateBucket(StorageRestStub):

        def __hash__(self):
            return hash('CreateBucket')

        def __call__(self, request: storage.CreateBucketRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Bucket:
            raise NotImplementedError('Method CreateBucket is not available over REST transport')

    class _CreateHmacKey(StorageRestStub):

        def __hash__(self):
            return hash('CreateHmacKey')

        def __call__(self, request: storage.CreateHmacKeyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.CreateHmacKeyResponse:
            raise NotImplementedError('Method CreateHmacKey is not available over REST transport')

    class _CreateNotificationConfig(StorageRestStub):

        def __hash__(self):
            return hash('CreateNotificationConfig')

        def __call__(self, request: storage.CreateNotificationConfigRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.NotificationConfig:
            raise NotImplementedError('Method CreateNotificationConfig is not available over REST transport')

    class _DeleteBucket(StorageRestStub):

        def __hash__(self):
            return hash('DeleteBucket')

        def __call__(self, request: storage.DeleteBucketRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            raise NotImplementedError('Method DeleteBucket is not available over REST transport')

    class _DeleteHmacKey(StorageRestStub):

        def __hash__(self):
            return hash('DeleteHmacKey')

        def __call__(self, request: storage.DeleteHmacKeyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            raise NotImplementedError('Method DeleteHmacKey is not available over REST transport')

    class _DeleteNotificationConfig(StorageRestStub):

        def __hash__(self):
            return hash('DeleteNotificationConfig')

        def __call__(self, request: storage.DeleteNotificationConfigRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            raise NotImplementedError('Method DeleteNotificationConfig is not available over REST transport')

    class _DeleteObject(StorageRestStub):

        def __hash__(self):
            return hash('DeleteObject')

        def __call__(self, request: storage.DeleteObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            raise NotImplementedError('Method DeleteObject is not available over REST transport')

    class _GetBucket(StorageRestStub):

        def __hash__(self):
            return hash('GetBucket')

        def __call__(self, request: storage.GetBucketRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Bucket:
            raise NotImplementedError('Method GetBucket is not available over REST transport')

    class _GetHmacKey(StorageRestStub):

        def __hash__(self):
            return hash('GetHmacKey')

        def __call__(self, request: storage.GetHmacKeyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.HmacKeyMetadata:
            raise NotImplementedError('Method GetHmacKey is not available over REST transport')

    class _GetIamPolicy(StorageRestStub):

        def __hash__(self):
            return hash('GetIamPolicy')

        def __call__(self, request: iam_policy_pb2.GetIamPolicyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> policy_pb2.Policy:
            raise NotImplementedError('Method GetIamPolicy is not available over REST transport')

    class _GetNotificationConfig(StorageRestStub):

        def __hash__(self):
            return hash('GetNotificationConfig')

        def __call__(self, request: storage.GetNotificationConfigRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.NotificationConfig:
            raise NotImplementedError('Method GetNotificationConfig is not available over REST transport')

    class _GetObject(StorageRestStub):

        def __hash__(self):
            return hash('GetObject')

        def __call__(self, request: storage.GetObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Object:
            raise NotImplementedError('Method GetObject is not available over REST transport')

    class _GetServiceAccount(StorageRestStub):

        def __hash__(self):
            return hash('GetServiceAccount')

        def __call__(self, request: storage.GetServiceAccountRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.ServiceAccount:
            raise NotImplementedError('Method GetServiceAccount is not available over REST transport')

    class _ListBuckets(StorageRestStub):

        def __hash__(self):
            return hash('ListBuckets')

        def __call__(self, request: storage.ListBucketsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.ListBucketsResponse:
            raise NotImplementedError('Method ListBuckets is not available over REST transport')

    class _ListHmacKeys(StorageRestStub):

        def __hash__(self):
            return hash('ListHmacKeys')

        def __call__(self, request: storage.ListHmacKeysRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.ListHmacKeysResponse:
            raise NotImplementedError('Method ListHmacKeys is not available over REST transport')

    class _ListNotificationConfigs(StorageRestStub):

        def __hash__(self):
            return hash('ListNotificationConfigs')

        def __call__(self, request: storage.ListNotificationConfigsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.ListNotificationConfigsResponse:
            raise NotImplementedError('Method ListNotificationConfigs is not available over REST transport')

    class _ListObjects(StorageRestStub):

        def __hash__(self):
            return hash('ListObjects')

        def __call__(self, request: storage.ListObjectsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.ListObjectsResponse:
            raise NotImplementedError('Method ListObjects is not available over REST transport')

    class _LockBucketRetentionPolicy(StorageRestStub):

        def __hash__(self):
            return hash('LockBucketRetentionPolicy')

        def __call__(self, request: storage.LockBucketRetentionPolicyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Bucket:
            raise NotImplementedError('Method LockBucketRetentionPolicy is not available over REST transport')

    class _QueryWriteStatus(StorageRestStub):

        def __hash__(self):
            return hash('QueryWriteStatus')

        def __call__(self, request: storage.QueryWriteStatusRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.QueryWriteStatusResponse:
            raise NotImplementedError('Method QueryWriteStatus is not available over REST transport')

    class _ReadObject(StorageRestStub):

        def __hash__(self):
            return hash('ReadObject')

        def __call__(self, request: storage.ReadObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> rest_streaming.ResponseIterator:
            raise NotImplementedError('Method ReadObject is not available over REST transport')

    class _RestoreObject(StorageRestStub):

        def __hash__(self):
            return hash('RestoreObject')

        def __call__(self, request: storage.RestoreObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Object:
            raise NotImplementedError('Method RestoreObject is not available over REST transport')

    class _RewriteObject(StorageRestStub):

        def __hash__(self):
            return hash('RewriteObject')

        def __call__(self, request: storage.RewriteObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.RewriteResponse:
            raise NotImplementedError('Method RewriteObject is not available over REST transport')

    class _SetIamPolicy(StorageRestStub):

        def __hash__(self):
            return hash('SetIamPolicy')

        def __call__(self, request: iam_policy_pb2.SetIamPolicyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> policy_pb2.Policy:
            raise NotImplementedError('Method SetIamPolicy is not available over REST transport')

    class _StartResumableWrite(StorageRestStub):

        def __hash__(self):
            return hash('StartResumableWrite')

        def __call__(self, request: storage.StartResumableWriteRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.StartResumableWriteResponse:
            raise NotImplementedError('Method StartResumableWrite is not available over REST transport')

    class _TestIamPermissions(StorageRestStub):

        def __hash__(self):
            return hash('TestIamPermissions')

        def __call__(self, request: iam_policy_pb2.TestIamPermissionsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> iam_policy_pb2.TestIamPermissionsResponse:
            raise NotImplementedError('Method TestIamPermissions is not available over REST transport')

    class _UpdateBucket(StorageRestStub):

        def __hash__(self):
            return hash('UpdateBucket')

        def __call__(self, request: storage.UpdateBucketRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Bucket:
            raise NotImplementedError('Method UpdateBucket is not available over REST transport')

    class _UpdateHmacKey(StorageRestStub):

        def __hash__(self):
            return hash('UpdateHmacKey')

        def __call__(self, request: storage.UpdateHmacKeyRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.HmacKeyMetadata:
            raise NotImplementedError('Method UpdateHmacKey is not available over REST transport')

    class _UpdateObject(StorageRestStub):

        def __hash__(self):
            return hash('UpdateObject')

        def __call__(self, request: storage.UpdateObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.Object:
            raise NotImplementedError('Method UpdateObject is not available over REST transport')

    class _WriteObject(StorageRestStub):

        def __hash__(self):
            return hash('WriteObject')

        def __call__(self, request: storage.WriteObjectRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> storage.WriteObjectResponse:
            raise NotImplementedError('Method WriteObject is not available over REST transport')

    @property
    def bidi_write_object(self) -> Callable[[storage.BidiWriteObjectRequest], storage.BidiWriteObjectResponse]:
        return self._BidiWriteObject(self._session, self._host, self._interceptor)

    @property
    def cancel_resumable_write(self) -> Callable[[storage.CancelResumableWriteRequest], storage.CancelResumableWriteResponse]:
        return self._CancelResumableWrite(self._session, self._host, self._interceptor)

    @property
    def compose_object(self) -> Callable[[storage.ComposeObjectRequest], storage.Object]:
        return self._ComposeObject(self._session, self._host, self._interceptor)

    @property
    def create_bucket(self) -> Callable[[storage.CreateBucketRequest], storage.Bucket]:
        return self._CreateBucket(self._session, self._host, self._interceptor)

    @property
    def create_hmac_key(self) -> Callable[[storage.CreateHmacKeyRequest], storage.CreateHmacKeyResponse]:
        return self._CreateHmacKey(self._session, self._host, self._interceptor)

    @property
    def create_notification_config(self) -> Callable[[storage.CreateNotificationConfigRequest], storage.NotificationConfig]:
        return self._CreateNotificationConfig(self._session, self._host, self._interceptor)

    @property
    def delete_bucket(self) -> Callable[[storage.DeleteBucketRequest], empty_pb2.Empty]:
        return self._DeleteBucket(self._session, self._host, self._interceptor)

    @property
    def delete_hmac_key(self) -> Callable[[storage.DeleteHmacKeyRequest], empty_pb2.Empty]:
        return self._DeleteHmacKey(self._session, self._host, self._interceptor)

    @property
    def delete_notification_config(self) -> Callable[[storage.DeleteNotificationConfigRequest], empty_pb2.Empty]:
        return self._DeleteNotificationConfig(self._session, self._host, self._interceptor)

    @property
    def delete_object(self) -> Callable[[storage.DeleteObjectRequest], empty_pb2.Empty]:
        return self._DeleteObject(self._session, self._host, self._interceptor)

    @property
    def get_bucket(self) -> Callable[[storage.GetBucketRequest], storage.Bucket]:
        return self._GetBucket(self._session, self._host, self._interceptor)

    @property
    def get_hmac_key(self) -> Callable[[storage.GetHmacKeyRequest], storage.HmacKeyMetadata]:
        return self._GetHmacKey(self._session, self._host, self._interceptor)

    @property
    def get_iam_policy(self) -> Callable[[iam_policy_pb2.GetIamPolicyRequest], policy_pb2.Policy]:
        return self._GetIamPolicy(self._session, self._host, self._interceptor)

    @property
    def get_notification_config(self) -> Callable[[storage.GetNotificationConfigRequest], storage.NotificationConfig]:
        return self._GetNotificationConfig(self._session, self._host, self._interceptor)

    @property
    def get_object(self) -> Callable[[storage.GetObjectRequest], storage.Object]:
        return self._GetObject(self._session, self._host, self._interceptor)

    @property
    def get_service_account(self) -> Callable[[storage.GetServiceAccountRequest], storage.ServiceAccount]:
        return self._GetServiceAccount(self._session, self._host, self._interceptor)

    @property
    def list_buckets(self) -> Callable[[storage.ListBucketsRequest], storage.ListBucketsResponse]:
        return self._ListBuckets(self._session, self._host, self._interceptor)

    @property
    def list_hmac_keys(self) -> Callable[[storage.ListHmacKeysRequest], storage.ListHmacKeysResponse]:
        return self._ListHmacKeys(self._session, self._host, self._interceptor)

    @property
    def list_notification_configs(self) -> Callable[[storage.ListNotificationConfigsRequest], storage.ListNotificationConfigsResponse]:
        return self._ListNotificationConfigs(self._session, self._host, self._interceptor)

    @property
    def list_objects(self) -> Callable[[storage.ListObjectsRequest], storage.ListObjectsResponse]:
        return self._ListObjects(self._session, self._host, self._interceptor)

    @property
    def lock_bucket_retention_policy(self) -> Callable[[storage.LockBucketRetentionPolicyRequest], storage.Bucket]:
        return self._LockBucketRetentionPolicy(self._session, self._host, self._interceptor)

    @property
    def query_write_status(self) -> Callable[[storage.QueryWriteStatusRequest], storage.QueryWriteStatusResponse]:
        return self._QueryWriteStatus(self._session, self._host, self._interceptor)

    @property
    def read_object(self) -> Callable[[storage.ReadObjectRequest], storage.ReadObjectResponse]:
        return self._ReadObject(self._session, self._host, self._interceptor)

    @property
    def restore_object(self) -> Callable[[storage.RestoreObjectRequest], storage.Object]:
        return self._RestoreObject(self._session, self._host, self._interceptor)

    @property
    def rewrite_object(self) -> Callable[[storage.RewriteObjectRequest], storage.RewriteResponse]:
        return self._RewriteObject(self._session, self._host, self._interceptor)

    @property
    def set_iam_policy(self) -> Callable[[iam_policy_pb2.SetIamPolicyRequest], policy_pb2.Policy]:
        return self._SetIamPolicy(self._session, self._host, self._interceptor)

    @property
    def start_resumable_write(self) -> Callable[[storage.StartResumableWriteRequest], storage.StartResumableWriteResponse]:
        return self._StartResumableWrite(self._session, self._host, self._interceptor)

    @property
    def test_iam_permissions(self) -> Callable[[iam_policy_pb2.TestIamPermissionsRequest], iam_policy_pb2.TestIamPermissionsResponse]:
        return self._TestIamPermissions(self._session, self._host, self._interceptor)

    @property
    def update_bucket(self) -> Callable[[storage.UpdateBucketRequest], storage.Bucket]:
        return self._UpdateBucket(self._session, self._host, self._interceptor)

    @property
    def update_hmac_key(self) -> Callable[[storage.UpdateHmacKeyRequest], storage.HmacKeyMetadata]:
        return self._UpdateHmacKey(self._session, self._host, self._interceptor)

    @property
    def update_object(self) -> Callable[[storage.UpdateObjectRequest], storage.Object]:
        return self._UpdateObject(self._session, self._host, self._interceptor)

    @property
    def write_object(self) -> Callable[[storage.WriteObjectRequest], storage.WriteObjectResponse]:
        return self._WriteObject(self._session, self._host, self._interceptor)

    @property
    def kind(self) -> str:
        return 'rest'

    def close(self):
        self._session.close()