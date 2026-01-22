import logging
import os
from functools import partial
import ipykernel
import jupyter_client.session as session
import param
from bokeh.document.events import MessageSentEvent
from bokeh.document.json import Literal, MessageSent, TypedDict
from bokeh.util.serialization import make_id
from ipykernel.comm import Comm, CommManager
from ipykernel.kernelbase import Kernel
from ipywidgets import Widget
from ipywidgets._version import __protocol_version__
from ipywidgets.widgets.widget import _remove_buffers
from ipywidgets_bokeh.kernel import (
from ipywidgets_bokeh.widget import IPyWidget
from tornado.ioloop import IOLoop
from traitlets import Any
from ..config import __version__
from ..util import classproperty
from .state import set_curdoc, state
class PanelSessionWebsocket(SessionWebsocket):

    def __init__(self, *args, **kwargs):
        session.Session.__init__(self, *args, **kwargs)
        self._document = kwargs.pop('document', None)
        self._queue = []
        self._document.on_message('ipywidgets_bokeh', self.receive)

    def send(self, stream, msg_type, content=None, parent=None, ident=None, buffers=None, track=False, header=None, metadata=None):
        msg = self.msg(msg_type, content=content, parent=parent, header=header, metadata=metadata)
        try:
            msg['channel'] = stream.channel
        except Exception:
            return
        packed = self.pack(msg)
        if buffers is not None and len(buffers) != 0:
            buffers = [packed] + buffers
            nbufs = len(buffers)
            start = 4 * (1 + nbufs)
            offsets = [start]
            for buffer in buffers[:-1]:
                start += memoryview(buffer).nbytes
                offsets.append(start)
            u32 = lambda n: n.to_bytes(4, 'big')
            items = [u32(nbufs)] + [u32(offset) for offset in offsets] + buffers
            data = b''.join(items)
        else:
            data = packed.decode('utf-8')
        event = MessageSentEventPatched(self._document, 'ipywidgets_bokeh', data)
        self._queue.append(event)
        self._document.add_next_tick_callback(self._dispatch)

    def _dispatch(self):
        try:
            for event in self._queue:
                self._document.callbacks.trigger_on_change(event)
        except Exception as e:
            param.main.param.warning(f'ipywidgets event dispatch failed with: {e}')
        finally:
            self._queue = []