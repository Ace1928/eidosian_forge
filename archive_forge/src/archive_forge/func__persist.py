from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def _persist(self, data: Dict[str, Any]) -> None:
    """Saves documents and embeddings to BigQuery."""
    from google.cloud import bigquery
    data_len = len(data[list(data.keys())[0]])
    if data_len == 0:
        return
    list_of_dicts = [dict(zip(data, t)) for t in zip(*data.values())]
    job_config = bigquery.LoadJobConfig()
    job_config.schema = self.vectors_table.schema
    job_config.schema_update_options = bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job = self.bq_client.load_table_from_json(list_of_dicts, self.vectors_table, job_config=job_config)
    job.result()