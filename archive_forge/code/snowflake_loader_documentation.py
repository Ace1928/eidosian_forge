from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Initialize Snowflake document loader.

        Args:
            query: The query to run in Snowflake.
            user: Snowflake user.
            password: Snowflake password.
            account: Snowflake account.
            warehouse: Snowflake warehouse.
            role: Snowflake role.
            database: Snowflake database
            schema: Snowflake schema
            parameters: Optional. Parameters to pass to the query.
            page_content_columns: Optional. Columns written to Document `page_content`.
            metadata_columns: Optional. Columns written to Document `metadata`.
        