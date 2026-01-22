from typing import Any, Dict, NoReturn, Pattern, Tuple, Type, TypeVar, Union
class LocalProtocolError(ProtocolError):

    def _reraise_as_remote_protocol_error(self) -> NoReturn:
        self.__class__ = RemoteProtocolError
        raise self