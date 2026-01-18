import asyncio
import copy
import io
import os
import sys
import IPython
import traitlets
import ipyvuetify as v
def vue_upload(self, content, buffers):
    listener_id = content['id']
    listener = self.chunk_listeners.get(listener_id)
    if listener:
        if listener.version != self.version:
            del self.chunk_listeners[listener_id]
        else:
            listener.handle_chunk(content, buffers[0])