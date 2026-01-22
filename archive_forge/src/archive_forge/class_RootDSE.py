import dataclasses
import socket
import ssl
import threading
import typing as t
@dataclasses.dataclass(frozen=True)
class RootDSE:
    default_naming_context: str
    subschema_subentry: str
    supported_controls: t.List[str]