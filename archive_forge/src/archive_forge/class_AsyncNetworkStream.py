import ssl
import time
import typing
class AsyncNetworkStream:

    async def read(self, max_bytes: int, timeout: typing.Optional[float]=None) -> bytes:
        raise NotImplementedError()

    async def write(self, buffer: bytes, timeout: typing.Optional[float]=None) -> None:
        raise NotImplementedError()

    async def aclose(self) -> None:
        raise NotImplementedError()

    async def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: typing.Optional[str]=None, timeout: typing.Optional[float]=None) -> 'AsyncNetworkStream':
        raise NotImplementedError()

    def get_extra_info(self, info: str) -> typing.Any:
        return None