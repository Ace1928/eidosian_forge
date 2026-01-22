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
class AdaptationRestInterceptor:
    """Interceptor for Adaptation.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the AdaptationRestTransport.

    .. code-block:: python
        class MyCustomAdaptationInterceptor(AdaptationRestInterceptor):
            def pre_create_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_custom_class(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_custom_class(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_custom_classes(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_custom_classes(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_custom_class(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_custom_class(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_phrase_set(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_phrase_set(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = AdaptationRestTransport(interceptor=MyCustomAdaptationInterceptor())
        client = AdaptationClient(transport=transport)


    """

    def pre_create_custom_class(self, request: cloud_speech_adaptation.CreateCustomClassRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.CreateCustomClassRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_create_custom_class(self, response: resource.CustomClass) -> resource.CustomClass:
        """Post-rpc interceptor for create_custom_class

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_create_phrase_set(self, request: cloud_speech_adaptation.CreatePhraseSetRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.CreatePhraseSetRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_create_phrase_set(self, response: resource.PhraseSet) -> resource.PhraseSet:
        """Post-rpc interceptor for create_phrase_set

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_delete_custom_class(self, request: cloud_speech_adaptation.DeleteCustomClassRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.DeleteCustomClassRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def pre_delete_phrase_set(self, request: cloud_speech_adaptation.DeletePhraseSetRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.DeletePhraseSetRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def pre_get_custom_class(self, request: cloud_speech_adaptation.GetCustomClassRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.GetCustomClassRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_get_custom_class(self, response: resource.CustomClass) -> resource.CustomClass:
        """Post-rpc interceptor for get_custom_class

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_get_phrase_set(self, request: cloud_speech_adaptation.GetPhraseSetRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.GetPhraseSetRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_get_phrase_set(self, response: resource.PhraseSet) -> resource.PhraseSet:
        """Post-rpc interceptor for get_phrase_set

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_list_custom_classes(self, request: cloud_speech_adaptation.ListCustomClassesRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.ListCustomClassesRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_custom_classes

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_list_custom_classes(self, response: cloud_speech_adaptation.ListCustomClassesResponse) -> cloud_speech_adaptation.ListCustomClassesResponse:
        """Post-rpc interceptor for list_custom_classes

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_list_phrase_set(self, request: cloud_speech_adaptation.ListPhraseSetRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.ListPhraseSetRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_list_phrase_set(self, response: cloud_speech_adaptation.ListPhraseSetResponse) -> cloud_speech_adaptation.ListPhraseSetResponse:
        """Post-rpc interceptor for list_phrase_set

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_update_custom_class(self, request: cloud_speech_adaptation.UpdateCustomClassRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.UpdateCustomClassRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_custom_class

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_update_custom_class(self, response: resource.CustomClass) -> resource.CustomClass:
        """Post-rpc interceptor for update_custom_class

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_update_phrase_set(self, request: cloud_speech_adaptation.UpdatePhraseSetRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_speech_adaptation.UpdatePhraseSetRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_phrase_set

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_update_phrase_set(self, response: resource.PhraseSet) -> resource.PhraseSet:
        """Post-rpc interceptor for update_phrase_set

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_get_operation(self, request: operations_pb2.GetOperationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[operations_pb2.GetOperationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_get_operation(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response

    def pre_list_operations(self, request: operations_pb2.ListOperationsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[operations_pb2.ListOperationsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Adaptation server.
        """
        return (request, metadata)

    def post_list_operations(self, response: operations_pb2.ListOperationsResponse) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
        return response