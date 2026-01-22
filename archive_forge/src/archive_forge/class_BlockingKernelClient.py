from __future__ import annotations
import typing as t
from traitlets import Type
from ..channels import HBChannel, ZMQSocketChannel
from ..client import KernelClient, reqrep
from ..utils import run_sync
class BlockingKernelClient(KernelClient):
    """A KernelClient with blocking APIs

    ``get_[channel]_msg()`` methods wait for and return messages on channels,
    raising :exc:`queue.Empty` if no message arrives within ``timeout`` seconds.
    """
    get_shell_msg = run_sync(KernelClient._async_get_shell_msg)
    get_iopub_msg = run_sync(KernelClient._async_get_iopub_msg)
    get_stdin_msg = run_sync(KernelClient._async_get_stdin_msg)
    get_control_msg = run_sync(KernelClient._async_get_control_msg)
    wait_for_ready = run_sync(KernelClient._async_wait_for_ready)
    shell_channel_class = Type(ZMQSocketChannel)
    iopub_channel_class = Type(ZMQSocketChannel)
    stdin_channel_class = Type(ZMQSocketChannel)
    hb_channel_class = Type(HBChannel)
    control_channel_class = Type(ZMQSocketChannel)
    _recv_reply = run_sync(KernelClient._async_recv_reply)
    execute = reqrep(wrapped, KernelClient.execute)
    history = reqrep(wrapped, KernelClient.history)
    complete = reqrep(wrapped, KernelClient.complete)
    inspect = reqrep(wrapped, KernelClient.inspect)
    kernel_info = reqrep(wrapped, KernelClient.kernel_info)
    comm_info = reqrep(wrapped, KernelClient.comm_info)
    is_alive = run_sync(KernelClient._async_is_alive)
    execute_interactive = run_sync(KernelClient._async_execute_interactive)
    shutdown = reqrep(wrapped, KernelClient.shutdown, channel='control')