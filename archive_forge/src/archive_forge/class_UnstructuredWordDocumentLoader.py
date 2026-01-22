import os
import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
class UnstructuredWordDocumentLoader(UnstructuredFileLoader):
    """Load `Microsoft Word` file using `Unstructured`.

    Works with both .docx and .doc files.
    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    loader = UnstructuredWordDocumentLoader(
        "example.docx", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-docx
    """

    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.file_utils.filetype import FileType, detect_filetype
        unstructured_version = tuple([int(x) for x in __unstructured_version__.split('.')])
        try:
            import magic
            is_doc = detect_filetype(self.file_path) == FileType.DOC
        except ImportError:
            _, extension = os.path.splitext(str(self.file_path))
            is_doc = extension == '.doc'
        if is_doc and unstructured_version < (0, 4, 11):
            raise ValueError(f'You are on unstructured version {__unstructured_version__}. Partitioning .doc files is only supported in unstructured>=0.4.11. Please upgrade the unstructured package and try again.')
        if is_doc:
            from unstructured.partition.doc import partition_doc
            return partition_doc(filename=self.file_path, **self.unstructured_kwargs)
        else:
            from unstructured.partition.docx import partition_docx
            return partition_docx(filename=self.file_path, **self.unstructured_kwargs)