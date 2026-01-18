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
def update_dataset_tag(self, *, dataset_id: Optional[ID_TYPE]=None, dataset_name: Optional[str]=None, as_of: datetime.datetime, tag: str) -> None:
    """Update the tags of a dataset.

        If the tag is already assigned to a different version of this dataset,
        the tag will be moved to the new version. The as_of parameter is used to
        determine which version of the dataset to apply the new tags to.
        It must be an exact version of the dataset to succeed. You can
        use the read_dataset_version method to find the exact version
        to apply the tags to.

        Parameters
        ----------
        dataset_id : UUID
            The ID of the dataset to update.
        as_of : datetime.datetime
            The timestamp of the dataset to apply the new tags to.
        tag : str
            The new tag to apply to the dataset.

        Examples:
        --------
        .. code-block:: python

            dataset_name = "my-dataset"
            # Get the version of a dataset <= a given timestamp
            dataset_version = client.read_dataset_version(
                dataset_name=dataset_name, as_of=datetime.datetime(2024, 1, 1)
            )
            # Assign that version a new tag
            client.update_dataset_tags(
                dataset_name="my-dataset",
                as_of=dataset_version.as_of,
                tag="prod",
            )
        """
    if dataset_name is not None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    if dataset_id is None:
        raise ValueError('Must provide either dataset name or ID')
    response = self.session.put(f'{self.api_url}/datasets/{_as_uuid(dataset_id, 'dataset_id')}/tags', headers=self._headers, json={'as_of': as_of.isoformat(), 'tag': tag})
    ls_utils.raise_for_status_with_text(response)