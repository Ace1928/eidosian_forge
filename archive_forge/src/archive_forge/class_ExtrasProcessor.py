import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
class ExtrasProcessor:
    """Processes data from extras files into service models."""

    def process(self, original_model, extra_models):
        """Processes data from a list of loaded extras files into a model

        :type original_model: dict
        :param original_model: The service model to load all the extras into.

        :type extra_models: iterable of dict
        :param extra_models: A list of loaded extras models.
        """
        for extras in extra_models:
            self._process(original_model, extras)

    def _process(self, model, extra_model):
        """Process a single extras model into a service model."""
        if 'merge' in extra_model:
            deep_merge(model, extra_model['merge'])