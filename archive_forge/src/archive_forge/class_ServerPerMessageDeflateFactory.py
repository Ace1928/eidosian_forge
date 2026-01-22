from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
class ServerPerMessageDeflateFactory(ServerExtensionFactory):
    """
    Server-side extension factory for the Per-Message Deflate extension.

    Parameters behave as described in `section 7.1 of RFC 7692`_.

    .. _section 7.1 of RFC 7692: https://www.rfc-editor.org/rfc/rfc7692.html#section-7.1

    Set them to :obj:`True` to include them in the negotiation offer without a
    value or to an integer value to include them with this value.

    Args:
        server_no_context_takeover: prevent server from using context takeover.
        client_no_context_takeover: prevent client from using context takeover.
        server_max_window_bits: maximum size of the server's LZ77 sliding window
            in bits, between 8 and 15.
        client_max_window_bits: maximum size of the client's LZ77 sliding window
            in bits, between 8 and 15.
        compress_settings: additional keyword arguments for :func:`zlib.compressobj`,
            excluding ``wbits``.
        require_client_max_window_bits: do not enable compression at all if
            client doesn't advertise support for ``client_max_window_bits``;
            the default behavior is to enable compression without enforcing
            ``client_max_window_bits``.

    """
    name = ExtensionName('permessage-deflate')

    def __init__(self, server_no_context_takeover: bool=False, client_no_context_takeover: bool=False, server_max_window_bits: Optional[int]=None, client_max_window_bits: Optional[int]=None, compress_settings: Optional[Dict[str, Any]]=None, require_client_max_window_bits: bool=False) -> None:
        """
        Configure the Per-Message Deflate extension factory.

        """
        if not (server_max_window_bits is None or 8 <= server_max_window_bits <= 15):
            raise ValueError('server_max_window_bits must be between 8 and 15')
        if not (client_max_window_bits is None or 8 <= client_max_window_bits <= 15):
            raise ValueError('client_max_window_bits must be between 8 and 15')
        if compress_settings is not None and 'wbits' in compress_settings:
            raise ValueError('compress_settings must not include wbits, set server_max_window_bits instead')
        if client_max_window_bits is None and require_client_max_window_bits:
            raise ValueError("require_client_max_window_bits is enabled, but client_max_window_bits isn't configured")
        self.server_no_context_takeover = server_no_context_takeover
        self.client_no_context_takeover = client_no_context_takeover
        self.server_max_window_bits = server_max_window_bits
        self.client_max_window_bits = client_max_window_bits
        self.compress_settings = compress_settings
        self.require_client_max_window_bits = require_client_max_window_bits

    def process_request_params(self, params: Sequence[ExtensionParameter], accepted_extensions: Sequence[Extension]) -> Tuple[List[ExtensionParameter], PerMessageDeflate]:
        """
        Process request parameters.

        Return response params and an extension instance.

        """
        if any((other.name == self.name for other in accepted_extensions)):
            raise exceptions.NegotiationError(f'skipped duplicate {self.name}')
        server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits = _extract_parameters(params, is_server=True)
        if self.server_no_context_takeover:
            if not server_no_context_takeover:
                server_no_context_takeover = True
        if self.client_no_context_takeover:
            if not client_no_context_takeover:
                client_no_context_takeover = True
        if self.server_max_window_bits is None:
            pass
        elif server_max_window_bits is None:
            server_max_window_bits = self.server_max_window_bits
        elif server_max_window_bits > self.server_max_window_bits:
            server_max_window_bits = self.server_max_window_bits
        if self.client_max_window_bits is None:
            if client_max_window_bits is True:
                client_max_window_bits = self.client_max_window_bits
        elif client_max_window_bits is None:
            if self.require_client_max_window_bits:
                raise exceptions.NegotiationError('required client_max_window_bits')
        elif client_max_window_bits is True:
            client_max_window_bits = self.client_max_window_bits
        elif self.client_max_window_bits < client_max_window_bits:
            client_max_window_bits = self.client_max_window_bits
        return (_build_parameters(server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits), PerMessageDeflate(client_no_context_takeover, server_no_context_takeover, client_max_window_bits or 15, server_max_window_bits or 15, self.compress_settings))