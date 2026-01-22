from __future__ import annotations
import os
from typing import (
@staticmethod
def _load_ontology_schema_from_file(local_file: str, local_file_format: str=None):
    """
        Parse the ontology schema statements from the provided file
        """
    import rdflib
    if not os.path.exists(local_file):
        raise FileNotFoundError(f'File {local_file} does not exist.')
    if not os.access(local_file, os.R_OK):
        raise PermissionError(f'Read permission for {local_file} is restricted')
    graph = rdflib.ConjunctiveGraph()
    try:
        graph.parse(local_file, format=local_file_format)
    except Exception as e:
        raise ValueError(f'Invalid file format for {local_file} : ', e)
    return graph