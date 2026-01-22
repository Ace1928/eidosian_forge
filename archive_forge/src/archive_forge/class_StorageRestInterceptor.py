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
class StorageRestInterceptor:
    """Interceptor for Storage.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the StorageRestTransport.

    .. code-block:: python
        class MyCustomStorageInterceptor(StorageRestInterceptor):
            def pre_cancel_resumable_write(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_cancel_resumable_write(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_compose_object(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_compose_object(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_bucket(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_hmac_key(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_hmac_key(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_notification_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_notification_config(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_hmac_key(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_notification_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_object(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_bucket(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_hmac_key(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_hmac_key(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_notification_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_notification_config(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_object(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_object(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_service_account(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_service_account(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_buckets(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_buckets(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_hmac_keys(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_hmac_keys(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_notification_configs(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_notification_configs(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_objects(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_objects(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_lock_bucket_retention_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_lock_bucket_retention_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_query_write_status(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_query_write_status(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_read_object(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_read_object(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_restore_object(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_restore_object(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_rewrite_object(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_rewrite_object(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_set_iam_policy(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_set_iam_policy(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_start_resumable_write(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_start_resumable_write(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_test_iam_permissions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_test_iam_permissions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_bucket(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_hmac_key(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_hmac_key(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_object(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_object(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = StorageRestTransport(interceptor=MyCustomStorageInterceptor())
        client = StorageClient(transport=transport)


    """