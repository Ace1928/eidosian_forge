from hashlib import md5
from typing import Any, Dict, List, Optional
from langchain_core.utils import get_from_dict_or_env
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

        This method constructs nodes and relationships in the graph based on the
        provided GraphDocument objects.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the MENTIONS relationship.
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document metadata
        if available; otherwise it calculates the MD5 hash of `page_content`
        for merging process. Defaults to False.
        - baseEntityLabel (bool, optional): If True, each newly created node
        gets a secondary __Entity__ label, which is indexed and improves import
        speed and performance. Defaults to False.
        