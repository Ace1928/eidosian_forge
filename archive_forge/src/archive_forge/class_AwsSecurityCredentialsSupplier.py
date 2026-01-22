import abc
from dataclasses import dataclass
import hashlib
import hmac
import http.client as http_client
import json
import os
import posixpath
import re
from typing import Optional
import urllib
from urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
class AwsSecurityCredentialsSupplier(metaclass=abc.ABCMeta):
    """Base class for AWS security credential suppliers. This can be implemented with custom logic to retrieve
    AWS security credentials to exchange for a Google Cloud access token. The AWS external account credential does
    not cache the AWS security credentials, so caching logic should be added in the implementation.
    """

    @abc.abstractmethod
    def get_aws_security_credentials(self, context, request):
        """Returns the AWS security credentials for the requested context.

        .. warning: This is not cached by the calling Google credential, so caching logic should be implemented in the supplier.

        Args:
            context (google.auth.externalaccount.SupplierContext): The context object
                containing information about the requested audience and subject token type.
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Raises:
            google.auth.exceptions.RefreshError: If an error is encountered during
                security credential retrieval logic.

        Returns:
            AwsSecurityCredentials: The requested AWS security credentials.
        """
        raise NotImplementedError('')

    @abc.abstractmethod
    def get_aws_region(self, context, request):
        """Returns the AWS region for the requested context.

        Args:
            context (google.auth.externalaccount.SupplierContext): The context object
                containing information about the requested audience and subject token type.
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Raises:
            google.auth.exceptions.RefreshError: If an error is encountered during
                region retrieval logic.

        Returns:
            str: The AWS region.
        """
        raise NotImplementedError('')