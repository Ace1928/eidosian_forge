from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _fetch_table_schema(self, table_name: str) -> Dict[str, str]:
    """Fetch the schema of a specified table.

        Args:
            table_name: The name of the table for which to fetch the schema.

        Returns:
            A dictionary mapping column names to their data types.
        """
    response = self.glue_client.get_table(DatabaseName=self.database, Name=table_name)
    columns = response['Table']['StorageDescriptor']['Columns']
    return {col['Name']: col['Type'] for col in columns}