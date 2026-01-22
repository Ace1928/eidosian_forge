import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
def _get_head_node_ip(address: Optional[str]=None):
    """Get the head node ip from the ray address if possible

    Args:
        address: ray cluster address, e.g. "auto", "localhost:6379"

    Raises:
        click.UsageError if node ip could not be resolved
    """
    try:
        address = services.canonicalize_bootstrap_address_or_die(address)
        return address.split(':')[0]
    except (ConnectionError, ValueError) as e:
        raise click.UsageError(str(e))