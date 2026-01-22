import copy
import itertools
import json
import logging
import os
import time
from collections import Counter
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Set, Tuple
import boto3
import botocore
from packaging.version import Version
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.providers import _PROVIDER_PRETTY_NAMES
from ray.autoscaler._private.util import check_legacy_fields
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def _create_default_inbound_rules(sgids, extended_rules=None):
    if extended_rules is None:
        extended_rules = []
    intracluster_rules = _create_default_intracluster_inbound_rules(sgids)
    ssh_rules = _create_default_ssh_inbound_rules()
    merged_rules = itertools.chain(intracluster_rules, ssh_rules, extended_rules)
    return list(merged_rules)