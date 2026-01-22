import logging
import threading
import urllib
import warnings
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import requests
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.dashboard.utils import (
from ray.util.annotations import DeveloperAPI
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException, ServerUnavailable
def _raise_on_missing_output(self, resource: StateResource, api_response: dict):
    """Raise an exception when the API resopnse contains a missing output.

        Output can be missing if (1) Failures on some of data source queries (e.g.,
        `ray list tasks` queries all raylets, and if some of queries fail, it will
        contain missing output. If all queries fail, it will just fail). (2) Data
        is truncated because the output is too large.

        Args:
            resource: Resource names, i.e. 'jobs', 'actors', 'nodes',
                see `StateResource` for details.
            api_response: The dictionarified `ListApiResponse` or `SummaryApiResponse`.
        """
    warning_msgs = api_response.get('partial_failure_warning', None)
    if warning_msgs:
        raise RayStateApiException(f'Failed to retrieve all {resource.value} from the cluster becausethey are not reachable due to query failures to the data sources. To avoid raising an exception and allow having missing output, set `raise_on_missing_output=False`. ')
    total = api_response['total']
    num_after_truncation = api_response['num_after_truncation']
    if total != num_after_truncation:
        raise RayStateApiException(f'Failed to retrieve all {total} {resource.value} from the cluster because they are not reachable due to data truncation. It happens when the returned data is too large (> {num_after_truncation}) To avoid raising an exception and allow having missing output, set `raise_on_missing_output=False`. ')