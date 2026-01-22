from __future__ import annotations
import os
from typing import (
def _check_connectivity(self) -> None:
    """
        Executes a simple `ASK` query to check connectivity
        """
    try:
        self.graph.query('ASK { ?s ?p ?o }')
    except ValueError:
        raise ValueError("Could not query the provided endpoint. Please, check, if the value of the provided query_endpoint points to the right repository. If GraphDB is secured, please, make sure that the environment variables 'GRAPHDB_USERNAME' and 'GRAPHDB_PASSWORD' are set.")