import asyncio
import dataclasses
import json
import logging
import traceback
from random import sample
from typing import Iterator, Optional
import aiohttp.web
from aiohttp.web import Request, Response
from aiohttp.client import ClientResponse
import ray
import ray.dashboard.optional_utils as optional_utils
import ray.dashboard.consts as dashboard_consts
from ray.dashboard.datacenter import DataOrganizer
import ray.dashboard.utils as dashboard_utils
from ray._private.runtime_env.packaging import (
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.pydantic_models import (
from ray.dashboard.modules.job.utils import (
from ray.dashboard.modules.version import (

        Try to disperse as much as possible to select one of
        the `CANDIDATE_AGENT_NUMBER` agents to solve requests.
        the agents will not pop from `self._agents` unless
        it's dead. Saved in `self._agents` is the agent that was
        used before.
        Strategy:
            1. if the number of `self._agents` has reached
               `CANDIDATE_AGENT_NUMBER`, randomly select one agent from
               `self._agents`.
            2. if not, randomly select one agent from all available agents,
               it is possible that the selected one already exists in
               `self._agents`.
        