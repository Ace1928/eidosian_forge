from pathlib import Path
from typing import Any, List, Union
from langchain_community.document_loaders.unstructured import (
Load `TSV` files using `Unstructured`.

    Like other
    Unstructured loaders, UnstructuredTSVLoader can be used in both
    "single" and "elements" mode. If you use the loader in "elements"
    mode, the TSV file will be a single Unstructured Table element.
    If you use the loader in "elements" mode, an HTML representation
    of the table will be available in the "text_as_html" key in the
    document metadata.

    Examples
    --------
    from langchain_community.document_loaders.tsv import UnstructuredTSVLoader

    loader = UnstructuredTSVLoader("stanley-cups.tsv", mode="elements")
    docs = loader.load()
    