import abc
import json
import os
from typing import NamedTuple
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
class SubjectTokenSupplier(metaclass=abc.ABCMeta):
    """Base class for subject token suppliers. This can be implemented with custom logic to retrieve
    a subject token to exchange for a Google Cloud access token when using Workload or
    Workforce Identity Federation. The identity pool credential does not cache the subject token,
    so caching logic should be added in the implementation.
    """

    @abc.abstractmethod
    def get_subject_token(self, context, request):
        """Returns the requested subject token. The subject token must be valid.

        .. warning: This is not cached by the calling Google credential, so caching logic should be implemented in the supplier.

        Args:
            context (google.auth.externalaccount.SupplierContext): The context object
                containing information about the requested audience and subject token type.
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Raises:
            google.auth.exceptions.RefreshError: If an error is encountered during
                subject token retrieval logic.

        Returns:
            str: The requested subject token string.
        """
        raise NotImplementedError('')