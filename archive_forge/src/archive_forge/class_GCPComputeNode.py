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
class GCPComputeNode(GCPNode):
    """Abstraction around compute nodes"""
    NON_TERMINATED_STATUSES = {'PROVISIONING', 'STAGING', 'RUNNING'}
    TERMINATED_STATUSES = {'TERMINATED', 'SUSPENDED'}
    RUNNING_STATUSES = {'RUNNING'}
    STATUS_FIELD = 'status'

    def get_labels(self) -> dict:
        return self.get('labels', {})

    def get_external_ip(self) -> str:
        return self.get('networkInterfaces', [{}])[0].get('accessConfigs', [{}])[0].get('natIP', None)

    def get_internal_ip(self) -> str:
        return self.get('networkInterfaces', [{}])[0].get('networkIP')