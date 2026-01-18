from contextlib import contextmanager
import time
from typing import Any, Dict
import ray as real_ray
from ray.job_config import JobConfig
import ray.util.client.server.server as ray_client_server
from ray.util.client import ray
from ray._private.client_mode_hook import enable_client_mode, disable_client_hook
Utility for running test logic with and without a Ray client connection.

    If client_connect is True, will connect to Ray client in context.
    If client_connect is False, does nothing.

    How to use:
    Given a test of the following form:

    def test_<name>(args):
        <initialize a ray cluster>
        <use the ray cluster>

    Modify the test to

    @pytest.mark.parametrize("connect_to_client", [False, True])
    def test_<name>(args, connect_to_client)
    <initialize a ray cluster>
    with connect_to_client_or_not(connect_to_client):
        <use the ray cluster>

    Parameterize the argument connect over True, False to run the test with and
    without a Ray client connection.
    