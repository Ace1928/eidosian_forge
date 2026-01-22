from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _initialize_glue_client(self) -> Any:
    """Initialize the AWS Glue client.

        Returns:
            The initialized AWS Glue client.

        Raises:
            ValueError: If there is an issue with AWS session/client initialization.
        """
    try:
        import boto3
    except ImportError as e:
        raise ImportError('boto3 is required to use the GlueCatalogLoader. Please install it with `pip install boto3`.') from e
    try:
        session = boto3.Session(profile_name=self.profile_name) if self.profile_name else boto3.Session()
        return session.client('glue')
    except Exception as e:
        raise ValueError('Issue with AWS session/client initialization.') from e