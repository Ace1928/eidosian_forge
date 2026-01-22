import csv
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_community.document_loaders.unstructured import (
def __read_file(self, csvfile: TextIOWrapper) -> Iterator[Document]:
    csv_reader = csv.DictReader(csvfile, **self.csv_args)
    for i, row in enumerate(csv_reader):
        try:
            source = row[self.source_column] if self.source_column is not None else str(self.file_path)
        except KeyError:
            raise ValueError(f"Source column '{self.source_column}' not found in CSV file.")
        content = '\n'.join((f'{k.strip()}: {(v.strip() if v is not None else v)}' for k, v in row.items() if k not in self.metadata_columns))
        metadata = {'source': source, 'row': i}
        for col in self.metadata_columns:
            try:
                metadata[col] = row[col]
            except KeyError:
                raise ValueError(f"Metadata column '{col}' not found in CSV file.")
        yield Document(page_content=content, metadata=metadata)