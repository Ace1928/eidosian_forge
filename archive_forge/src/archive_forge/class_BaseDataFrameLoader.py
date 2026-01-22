from typing import Any, Iterator, Literal
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class BaseDataFrameLoader(BaseLoader):

    def __init__(self, data_frame: Any, *, page_content_column: str='text'):
        """Initialize with dataframe object.

        Args:
            data_frame: DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""
        for _, row in self.data_frame.iterrows():
            text = row[self.page_content_column]
            metadata = row.to_dict()
            metadata.pop(self.page_content_column)
            yield Document(page_content=text, metadata=metadata)