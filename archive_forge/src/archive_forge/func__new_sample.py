import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
def _new_sample(self, sink):
    """The callback for appsink's "new-sample" signal.
        """
    if self.running:
        buf = sink.emit('pull-sample').get_buffer()
        mem = buf.get_all_memory()
        success, info = mem.map(Gst.MapFlags.READ)
        if success:
            if isinstance(info.data, memoryview):
                data = bytes(info.data)
            else:
                data = info.data
            mem.unmap(info)
            self.queue.put(data)
        else:
            raise GStreamerError('Unable to map buffer memory while reading the file.')
    return Gst.FlowReturn.OK