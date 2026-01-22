import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders
class ClientConnectorCertificateError(*cert_errors_bases):
    """Response certificate error."""

    def __init__(self, connection_key: ConnectionKey, certificate_error: Exception) -> None:
        self._conn_key = connection_key
        self._certificate_error = certificate_error
        self.args = (connection_key, certificate_error)

    @property
    def certificate_error(self) -> Exception:
        return self._certificate_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> Optional[int]:
        return self._conn_key.port

    @property
    def ssl(self) -> bool:
        return self._conn_key.is_ssl

    def __str__(self) -> str:
        return 'Cannot connect to host {0.host}:{0.port} ssl:{0.ssl} [{0.certificate_error.__class__.__name__}: {0.certificate_error.args}]'.format(self)