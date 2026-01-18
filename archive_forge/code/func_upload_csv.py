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
def upload_csv(self, csv_file: Union[str, Tuple[str, io.BytesIO]], input_keys: Sequence[str], output_keys: Sequence[str], *, name: Optional[str]=None, description: Optional[str]=None, data_type: Optional[ls_schemas.DataType]=ls_schemas.DataType.kv) -> ls_schemas.Dataset:
    """Upload a CSV file to the LangSmith API.

        Parameters
        ----------
        csv_file : str or Tuple[str, BytesIO]
            The CSV file to upload. If a string, it should be the path
            If a tuple, it should be a tuple containing the filename
            and a BytesIO object.
        input_keys : Sequence[str]
            The input keys.
        output_keys : Sequence[str]
            The output keys.
        name : str or None, default=None
            The name of the dataset.
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
    data = {'input_keys': input_keys, 'output_keys': output_keys}
    if name:
        data['name'] = name
    if description:
        data['description'] = description
    if data_type:
        data['data_type'] = ls_utils.get_enum_value(data_type)
    if isinstance(csv_file, str):
        with open(csv_file, 'rb') as f:
            file_ = {'file': f}
            response = self.session.post(self.api_url + '/datasets/upload', headers=self._headers, data=data, files=file_)
    elif isinstance(csv_file, tuple):
        response = self.session.post(self.api_url + '/datasets/upload', headers=self._headers, data=data, files={'file': csv_file})
    else:
        raise ValueError('csv_file must be a string or tuple')
    ls_utils.raise_for_status_with_text(response)
    result = response.json()
    if 'detail' in result and 'already exists' in result['detail']:
        file_name = csv_file if isinstance(csv_file, str) else csv_file[0]
        file_name = file_name.split('/')[-1]
        raise ValueError(f'Dataset {file_name} already exists')
    return ls_schemas.Dataset(**result, _host_url=self._host_url, _tenant_id=self._get_optional_tenant_id())