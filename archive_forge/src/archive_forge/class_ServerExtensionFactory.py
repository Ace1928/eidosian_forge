from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from .. import frames
from ..typing import ExtensionName, ExtensionParameter
class ServerExtensionFactory:
    """
    Base class for server-side extension factories.

    """
    name: ExtensionName
    'Extension identifier.'

    def process_request_params(self, params: Sequence[ExtensionParameter], accepted_extensions: Sequence[Extension]) -> Tuple[List[ExtensionParameter], Extension]:
        """
        Process parameters received from the client.

        Args:
            params (Sequence[ExtensionParameter]): parameters received from
                the client for this extension.
            accepted_extensions (Sequence[Extension]): list of previously
                accepted extensions.

        Returns:
            Tuple[List[ExtensionParameter], Extension]: To accept the offer,
            parameters to send to the client for this extension and an
            extension instance.

        Raises:
            NegotiationError: to reject the offer, if parameters received from
                the client aren't acceptable.

        """
        raise NotImplementedError