from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def upload_dataframe(self, df: pd.DataFrame, name: str, input_keys: Sequence[str], output_keys: Sequence[str], *, description: Optional[str]=None, data_type: Optional[ls_schemas.DataType]=ls_schemas.DataType.kv) -> ls_schemas.Dataset:
    """Upload a dataframe as individual examples to the LangSmith API.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to upload.
        name : str
            The name of the dataset.
        input_keys : Sequence[str]
            The input keys.
        output_keys : Sequence[str]
            The output keys.
        description : str or None, default=None
            The description of the dataset.
        data_type : DataType or None, default=DataType.kv
            The data type of the dataset.

        Returns:
        -------
        Dataset
            The uploaded dataset.

        Raises:
        ------
        ValueError
            If the csv_file is not a string or tuple.
        """
    csv_file = io.BytesIO()
    df.to_csv(csv_file, index=False)
    csv_file.seek(0)
    return self.upload_csv(('data.csv', csv_file), input_keys=input_keys, output_keys=output_keys, description=description, name=name, data_type=data_type)