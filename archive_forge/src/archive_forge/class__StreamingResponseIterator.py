from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
class _StreamingResponseIterator(Generic[P], grpc.Call):

    def __init__(self, wrapped, prefetch_first_result=True):
        self._wrapped = wrapped
        try:
            if prefetch_first_result:
                self._stored_first_result = next(self._wrapped)
        except TypeError:
            pass
        except StopIteration:
            pass

    def __iter__(self) -> Iterator[P]:
        """This iterator is also an iterable that returns itself."""
        return self

    def __next__(self) -> P:
        """Get the next response from the stream.

        Returns:
            protobuf.Message: A single response from the stream.
        """
        try:
            if hasattr(self, '_stored_first_result'):
                result = self._stored_first_result
                del self._stored_first_result
                return result
            return next(self._wrapped)
        except grpc.RpcError as exc:
            raise exceptions.from_grpc_error(exc) from exc

    def add_callback(self, callback):
        return self._wrapped.add_callback(callback)

    def cancel(self):
        return self._wrapped.cancel()

    def code(self):
        return self._wrapped.code()

    def details(self):
        return self._wrapped.details()

    def initial_metadata(self):
        return self._wrapped.initial_metadata()

    def is_active(self):
        return self._wrapped.is_active()

    def time_remaining(self):
        return self._wrapped.time_remaining()

    def trailing_metadata(self):
        return self._wrapped.trailing_metadata()