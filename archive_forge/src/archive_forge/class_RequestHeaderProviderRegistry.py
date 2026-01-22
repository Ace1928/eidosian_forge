import logging
import warnings
import entrypoints
from mlflow.tracking.request_header.databricks_request_header_provider import (
from mlflow.tracking.request_header.default_request_header_provider import (
class RequestHeaderProviderRegistry:

    def __init__(self):
        self._registry = []

    def register(self, request_header_provider):
        self._registry.append(request_header_provider())

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all('mlflow.request_header_provider'):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn('Failure attempting to register request header provider "{}": {}'.format(entrypoint.name, str(exc)), stacklevel=2)

    def __iter__(self):
        return iter(self._registry)