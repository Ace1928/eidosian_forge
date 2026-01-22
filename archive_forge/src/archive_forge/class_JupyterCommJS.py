import os
import sys
import uuid
import traceback
import json
import param
from ._version import __version__
class JupyterCommJS(JupyterComm):
    """
    JupyterCommJS provides a comms channel for the Jupyter notebook,
    which is initialized on the frontend. This allows sending events
    initiated on the frontend to python.
    """
    js_template = '\n    <script>\n      function msg_handler(msg) {{\n        var msg = msg.content.data;\n        var buffers = msg.buffers\n        {msg_handler}\n      }}\n      var comm = window.PyViz.comm_manager.get_client_comm("{comm_id}");\n      comm.on_msg(msg_handler);\n    </script>\n    '

    @classmethod
    def decode(cls, msg):
        decoded = dict(msg['content']['data'])
        if 'buffers' in msg:
            decoded['_buffers'] = {i: v for i, v in enumerate(msg['buffers'])}
        return decoded

    def __init__(self, id=None, on_msg=None, on_error=None, on_stdout=None, on_open=None):
        """
        Initializes a Comms object
        """
        from IPython import get_ipython
        super(JupyterCommJS, self).__init__(id, on_msg, on_error, on_stdout, on_open)
        self.manager = get_ipython().kernel.comm_manager
        self.manager.register_target(self.id, self._handle_open)

    def close(self):
        """
        Closes the comm connection
        """
        if self._comm:
            self._comm.close()
        elif self.id in self.manager.targets:
            del self.manager.targets[self.id]
        else:
            raise AssertionError('JupyterCommJS %s is already closed' % self.id)

    def _handle_open(self, comm, msg):
        self._comm = comm
        self._comm.on_msg(self._handle_msg)
        if self._on_open:
            self._on_open(msg)

    def send(self, data=None, metadata=None, buffers=[]):
        """
        Pushes data across comm socket.
        """
        self.comm.send(data, metadata=metadata, buffers=buffers)