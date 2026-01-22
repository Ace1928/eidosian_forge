import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class ParquetSerializer(BaseSerializer):
    """Serialize data in `Apache Parquet` format using the `pyarrow` package."""

    def __init__(self, persist_path: str) -> None:
        super().__init__(persist_path)
        self.pd = guard_import('pandas')
        self.pa = guard_import('pyarrow')
        self.pq = guard_import('pyarrow.parquet')

    @classmethod
    def extension(cls) -> str:
        return 'parquet'

    def save(self, data: Any) -> None:
        df = self.pd.DataFrame(data)
        table = self.pa.Table.from_pandas(df)
        if os.path.exists(self.persist_path):
            backup_path = str(self.persist_path) + '-backup'
            os.rename(self.persist_path, backup_path)
            try:
                self.pq.write_table(table, self.persist_path)
            except Exception as exc:
                os.rename(backup_path, self.persist_path)
                raise exc
            else:
                os.remove(backup_path)
        else:
            self.pq.write_table(table, self.persist_path)

    def load(self) -> Any:
        table = self.pq.read_table(self.persist_path)
        df = table.to_pandas()
        return {col: series.tolist() for col, series in df.items()}