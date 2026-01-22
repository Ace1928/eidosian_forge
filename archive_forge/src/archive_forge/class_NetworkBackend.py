import ssl
import time
import typing
class NetworkBackend:

    def connect_tcp(self, host: str, port: int, timeout: typing.Optional[float]=None, local_address: typing.Optional[str]=None, socket_options: typing.Optional[typing.Iterable[SOCKET_OPTION]]=None) -> NetworkStream:
        raise NotImplementedError()

    def connect_unix_socket(self, path: str, timeout: typing.Optional[float]=None, socket_options: typing.Optional[typing.Iterable[SOCKET_OPTION]]=None) -> NetworkStream:
        raise NotImplementedError()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)