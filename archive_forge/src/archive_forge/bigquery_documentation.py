from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.vertexai import get_client_info
Initialize BigQuery document loader.

        Args:
            query: The query to run in BigQuery.
            project: Optional. The project to run the query in.
            page_content_columns: Optional. The columns to write into the `page_content`
                of the document.
            metadata_columns: Optional. The columns to write into the `metadata` of the
                document.
            credentials : google.auth.credentials.Credentials, optional
              Credentials for accessing Google APIs. Use this parameter to override
                default credentials, such as to use Compute Engine
                (`google.auth.compute_engine.Credentials`) or Service Account
                (`google.oauth2.service_account.Credentials`) credentials directly.
        