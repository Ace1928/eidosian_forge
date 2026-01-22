from API responses.
import abc
import logging
import re
import time
from collections import UserDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
class GCPNode(UserDict, metaclass=abc.ABCMeta):
    """Abstraction around compute and tpu nodes"""
    NON_TERMINATED_STATUSES = None
    RUNNING_STATUSES = None
    STATUS_FIELD = None

    def __init__(self, base_dict: dict, resource: 'GCPResource', **kwargs) -> None:
        super().__init__(base_dict, **kwargs)
        self.resource = resource
        assert isinstance(self.resource, GCPResource)

    def is_running(self) -> bool:
        return self.get(self.STATUS_FIELD) in self.RUNNING_STATUSES

    def is_terminated(self) -> bool:
        return self.get(self.STATUS_FIELD) not in self.NON_TERMINATED_STATUSES

    @abc.abstractmethod
    def get_labels(self) -> dict:
        return

    @abc.abstractmethod
    def get_external_ip(self) -> str:
        return

    @abc.abstractmethod
    def get_internal_ip(self) -> str:
        return

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.get('name')}>'