import asyncio
from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync
from traitlets import Instance, Type, default
from .channels import InProcessChannel, InProcessHBChannel
def _dispatch_to_kernel(self, msg):
    """Send a message to the kernel and handle a reply."""
    kernel = self.kernel
    if kernel is None:
        msg = 'Cannot send request. No kernel exists.'
        raise RuntimeError(msg)
    stream = kernel.shell_stream
    self.session.send(stream, msg)
    msg_parts = stream.recv_multipart()
    if run_sync is not None:
        dispatch_shell = run_sync(kernel.dispatch_shell)
        dispatch_shell(msg_parts)
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(kernel.dispatch_shell(msg_parts))
    idents, reply_msg = self.session.recv(stream, copy=False)
    self.shell_channel.call_handlers_later(reply_msg)