from typing import Tuple
from azure.core.exceptions import HttpResponseError  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from azure.storage.blob import BlobClient, BlobServiceClient  # type: ignore
from ..errors import LaunchError
from ..utils import AZURE_BLOB_REGEX
from .abstract import AbstractEnvironment
Parse an Azure blob storage URI into a storage account and container.

        Args:
            uri (str): The URI to parse.

        Returns:
            Tuple[str, str, prefix]: The storage account, container, and path.
        