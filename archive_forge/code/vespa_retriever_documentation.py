from __future__ import annotations
import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
Instantiate retriever from params.

        Args:
            url (str): Vespa app URL.
            content_field (str): Field in results to return as Document page_content.
            k (Optional[int]): Number of Documents to return. Defaults to None.
            metadata_fields(Sequence[str] or "*"): Fields in results to include in
                document metadata. Defaults to empty tuple ().
            sources (Sequence[str] or "*" or None): Sources to retrieve
                from. Defaults to None.
            _filter (Optional[str]): Document filter condition expressed in YQL.
                Defaults to None.
            yql (Optional[str]): Full YQL query to be used. Should not be specified
                if _filter or sources are specified. Defaults to None.
            kwargs (Any): Keyword arguments added to query body.

        Returns:
            VespaRetriever: Instantiated VespaRetriever.
        