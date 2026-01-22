from base64 import b64encode
import io
import json
import pathlib
import uuid
from ipykernel.comm import Comm
from IPython.display import display, Javascript, HTML
from matplotlib import is_interactive
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import _Backend, CloseEvent, NavigationToolbar2
from .backend_webagg_core import (
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
class FigureManagerNbAgg(FigureManagerWebAgg):
    _toolbar2_class = ToolbarCls = NavigationIPy

    def __init__(self, canvas, num):
        self._shown = False
        super().__init__(canvas, num)

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        canvas = canvas_class(figure)
        manager = cls(canvas, num)
        if is_interactive():
            manager.show()
            canvas.draw_idle()

        def destroy(event):
            canvas.mpl_disconnect(cid)
            Gcf.destroy(manager)
        cid = canvas.mpl_connect('close_event', destroy)
        return manager

    def display_js(self):
        display(Javascript(FigureManagerNbAgg.get_javascript()))

    def show(self):
        if not self._shown:
            self.display_js()
            self._create_comm()
        else:
            self.canvas.draw_idle()
        self._shown = True
        if hasattr(self, '_cidgcf'):
            self.canvas.mpl_disconnect(self._cidgcf)
        if not is_interactive():
            from matplotlib._pylab_helpers import Gcf
            Gcf.figs.pop(self.num, None)

    def reshow(self):
        """
        A special method to re-show the figure in the notebook.

        """
        self._shown = False
        self.show()

    @property
    def connected(self):
        return bool(self.web_sockets)

    @classmethod
    def get_javascript(cls, stream=None):
        if stream is None:
            output = io.StringIO()
        else:
            output = stream
        super().get_javascript(stream=output)
        output.write((pathlib.Path(__file__).parent / 'web_backend/js/nbagg_mpl.js').read_text(encoding='utf-8'))
        if stream is None:
            return output.getvalue()

    def _create_comm(self):
        comm = CommSocket(self)
        self.add_web_socket(comm)
        return comm

    def destroy(self):
        self._send_event('close')
        for comm in list(self.web_sockets):
            comm.on_close()
        self.clearup_closed()

    def clearup_closed(self):
        """Clear up any closed Comms."""
        self.web_sockets = {socket for socket in self.web_sockets if socket.is_open()}
        if len(self.web_sockets) == 0:
            CloseEvent('close_event', self.canvas)._process()

    def remove_comm(self, comm_id):
        self.web_sockets = {socket for socket in self.web_sockets if socket.comm.comm_id != comm_id}