from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _fetch_tables(self) -> List[str]:
    """Retrieve all table names in the specified Glue database.

        Returns:
            A list of table names.
        """
    paginator = self.glue_client.get_paginator('get_tables')
    table_names = []
    for page in paginator.paginate(DatabaseName=self.database):
        for table in page['TableList']:
            if self.table_filter is None or table['Name'] in self.table_filter:
                table_names.append(table['Name'])
    return table_names