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
from google.cloud.texttospeech_v1.types import cloud_tts
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import TextToSpeechTransport
class TextToSpeechRestInterceptor:
    """Interceptor for TextToSpeech.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the TextToSpeechRestTransport.

    .. code-block:: python
        class MyCustomTextToSpeechInterceptor(TextToSpeechRestInterceptor):
            def pre_list_voices(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_voices(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_synthesize_speech(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_synthesize_speech(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = TextToSpeechRestTransport(interceptor=MyCustomTextToSpeechInterceptor())
        client = TextToSpeechClient(transport=transport)


    """

    def pre_list_voices(self, request: cloud_tts.ListVoicesRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_tts.ListVoicesRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_voices

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TextToSpeech server.
        """
        return (request, metadata)

    def post_list_voices(self, response: cloud_tts.ListVoicesResponse) -> cloud_tts.ListVoicesResponse:
        """Post-rpc interceptor for list_voices

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeech server but before
        it is returned to user code.
        """
        return response

    def pre_synthesize_speech(self, request: cloud_tts.SynthesizeSpeechRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[cloud_tts.SynthesizeSpeechRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for synthesize_speech

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TextToSpeech server.
        """
        return (request, metadata)

    def post_synthesize_speech(self, response: cloud_tts.SynthesizeSpeechResponse) -> cloud_tts.SynthesizeSpeechResponse:
        """Post-rpc interceptor for synthesize_speech

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeech server but before
        it is returned to user code.
        """
        return response

    def pre_get_operation(self, request: operations_pb2.GetOperationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[operations_pb2.GetOperationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_operation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TextToSpeech server.
        """
        return (request, metadata)

    def post_get_operation(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for get_operation

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeech server but before
        it is returned to user code.
        """
        return response

    def pre_list_operations(self, request: operations_pb2.ListOperationsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[operations_pb2.ListOperationsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_operations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the TextToSpeech server.
        """
        return (request, metadata)

    def post_list_operations(self, response: operations_pb2.ListOperationsResponse) -> operations_pb2.ListOperationsResponse:
        """Post-rpc interceptor for list_operations

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeech server but before
        it is returned to user code.
        """
        return response