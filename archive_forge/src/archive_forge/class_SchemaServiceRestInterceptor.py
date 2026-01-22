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
from google.pubsub_v1.types import schema
from google.pubsub_v1.types import schema as gp_schema
from .base import (
class SchemaServiceRestInterceptor:
    """Interceptor for SchemaService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the SchemaServiceRestTransport.

    .. code-block:: python
        class MyCustomSchemaServiceInterceptor(SchemaServiceRestInterceptor):
            def pre_commit_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_commit_schema(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_schema(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_schema_revision(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_schema_revision(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_schema(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_schema_revisions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_schema_revisions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_schemas(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_schemas(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_rollback_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_rollback_schema(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_validate_message(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_validate_message(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_validate_schema(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_validate_schema(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = SchemaServiceRestTransport(interceptor=MyCustomSchemaServiceInterceptor())
        client = SchemaServiceClient(transport=transport)


    """

    def pre_commit_schema(self, request: gp_schema.CommitSchemaRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[gp_schema.CommitSchemaRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for commit_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_commit_schema(self, response: gp_schema.Schema) -> gp_schema.Schema:
        """Post-rpc interceptor for commit_schema

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_create_schema(self, request: gp_schema.CreateSchemaRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[gp_schema.CreateSchemaRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_create_schema(self, response: gp_schema.Schema) -> gp_schema.Schema:
        """Post-rpc interceptor for create_schema

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_delete_schema(self, request: schema.DeleteSchemaRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.DeleteSchemaRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def pre_delete_schema_revision(self, request: schema.DeleteSchemaRevisionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.DeleteSchemaRevisionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_schema_revision

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_delete_schema_revision(self, response: schema.Schema) -> schema.Schema:
        """Post-rpc interceptor for delete_schema_revision

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_get_schema(self, request: schema.GetSchemaRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.GetSchemaRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_get_schema(self, response: schema.Schema) -> schema.Schema:
        """Post-rpc interceptor for get_schema

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_list_schema_revisions(self, request: schema.ListSchemaRevisionsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.ListSchemaRevisionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_schema_revisions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_list_schema_revisions(self, response: schema.ListSchemaRevisionsResponse) -> schema.ListSchemaRevisionsResponse:
        """Post-rpc interceptor for list_schema_revisions

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_list_schemas(self, request: schema.ListSchemasRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.ListSchemasRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_schemas

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_list_schemas(self, response: schema.ListSchemasResponse) -> schema.ListSchemasResponse:
        """Post-rpc interceptor for list_schemas

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_rollback_schema(self, request: schema.RollbackSchemaRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.RollbackSchemaRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for rollback_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_rollback_schema(self, response: schema.Schema) -> schema.Schema:
        """Post-rpc interceptor for rollback_schema

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_validate_message(self, request: schema.ValidateMessageRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.ValidateMessageRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for validate_message

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_validate_message(self, response: schema.ValidateMessageResponse) -> schema.ValidateMessageResponse:
        """Post-rpc interceptor for validate_message

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_validate_schema(self, request: gp_schema.ValidateSchemaRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[gp_schema.ValidateSchemaRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for validate_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_validate_schema(self, response: gp_schema.ValidateSchemaResponse) -> gp_schema.ValidateSchemaResponse:
        """Post-rpc interceptor for validate_schema

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_get_iam_policy(self, request: iam_policy_pb2.GetIamPolicyRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[iam_policy_pb2.GetIamPolicyRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_get_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_set_iam_policy(self, request: iam_policy_pb2.SetIamPolicyRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[iam_policy_pb2.SetIamPolicyRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_set_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response

    def pre_test_iam_permissions(self, request: iam_policy_pb2.TestIamPermissionsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[iam_policy_pb2.TestIamPermissionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
        return (request, metadata)

    def post_test_iam_permissions(self, response: iam_policy_pb2.TestIamPermissionsResponse) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the response
        after it is returned by the SchemaService server but before
        it is returned to user code.
        """
        return response