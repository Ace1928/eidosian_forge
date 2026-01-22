import logging
from typing import TYPE_CHECKING, Any, Optional
from ..lib.mailbox import Mailbox
from ..lib.sock_client import SockClient
from .interface_shared import InterfaceShared
from .message_future import MessageFuture
from .router_sock import MessageSockRouter
def _communicate_async(self, rec: 'pb.Record', local: Optional[bool]=None) -> MessageFuture:
    self._assign(rec)
    assert self._router
    if self._process_check and self._process and (not self._process.is_alive()):
        raise Exception('The wandb backend process has shutdown')
    future = self._router.send_and_receive(rec, local=local)
    return future