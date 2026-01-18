import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils

        Update the Swarm's configuration

        Args:
            version (int): The version number of the swarm object being
                updated. This is required to avoid conflicting writes.
            swarm_spec (dict): Configuration settings to update. Use
                :py:meth:`~docker.api.swarm.SwarmApiMixin.create_swarm_spec` to
                generate a valid configuration. Default: ``None``.
            rotate_worker_token (bool): Rotate the worker join token. Default:
                ``False``.
            rotate_manager_token (bool): Rotate the manager join token.
                Default: ``False``.
            rotate_manager_unlock_key (bool): Rotate the manager unlock key.
                Default: ``False``.

        Returns:
            ``True`` if the request went through.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        