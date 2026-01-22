from typing import Dict, List, Optional, cast
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


        Args:
            query: The query to execute.
            database: The database to connect to. Defaults to ":memory:".
            read_only: Whether to open the database in read-only mode.
              Defaults to False.
            config: A dictionary of configuration options to pass to the database.
              Optional.
            page_content_columns: The columns to write into the `page_content`
              of the document. Optional.
            metadata_columns: The columns to write into the `metadata` of the document.
              Optional.
        