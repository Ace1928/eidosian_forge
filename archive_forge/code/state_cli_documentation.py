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
Try resolve the command line args assuming users omitted the subcommand.

        This overrides the default `resolve_command` for the parent class.
        This will allow command alias of `ray <glob>` to `ray cluster <glob>`.
        