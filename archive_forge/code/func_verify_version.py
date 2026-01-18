from __future__ import annotations
import enum
import logging
import os
from hashlib import md5
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def verify_version(self) -> None:
    """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
    db_data = self.query('CALL dbms.components()')
    version = db_data[0]['versions'][0]
    if 'aura' in version:
        version_tuple = tuple(map(int, version.split('-')[0].split('.'))) + (0,)
    else:
        version_tuple = tuple(map(int, version.split('.')))
    target_version = (5, 11, 0)
    if version_tuple < target_version:
        raise ValueError('Version index is only supported in Neo4j version 5.11 or greater')
    metadata_target_version = (5, 18, 0)
    if version_tuple < metadata_target_version:
        self.support_metadata_filter = False
    else:
        self.support_metadata_filter = True
    self._is_enterprise = True if db_data[0]['edition'] == 'enterprise' else False