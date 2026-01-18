from .. import errors
from .resource import Collection, Model

        List plugins installed on the server.

        Returns:
            (list of :py:class:`Plugin`): The plugins.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        