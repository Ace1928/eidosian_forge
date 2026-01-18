import datetime
import sys
import time
import urllib
import platform
from typing import Any, Callable, cast, Dict, List, Optional
import json.decoder as jd
import requests
import cirq_ionq
from cirq_ionq import ionq_exceptions
from cirq import __version__ as cirq_version
Helper method for list calls.

        Args:
            resource_path: The resource path for the object being listed. Follows the base url
                and version. No leading slash.
            params: The params to pass with the list call.
            response_key: The key to get the list of objects that have been listed.
            limit: The maximum number of objects to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A sequence of dictionaries corresponding to the objects listed.
        