from __future__ import division
import numpy as np
from pygsp import utils
def _qtg_plot_graph(G, show_edges, vertex_size, plot_name):
    qtg, gl, QtGui = _import_qtg()
    if G.is_directed():
        raise NotImplementedError
    elif G.coords.shape[1] == 2:
        window = qtg.GraphicsWindow()
        window.setWindowTitle(plot_name)
        view = window.addViewBox()
        view.setAspectLocked()
        if show_edges:
            pen = tuple(np.array(G.plotting['edge_color']) * 255)
        else:
            pen = None
        adj = _get_coords(G, edge_list=True)
        g = qtg.GraphItem(pos=G.coords, adj=adj, pen=pen, size=vertex_size / 10)
        view.addItem(g)
        global _qtg_windows
        _qtg_windows.append(window)
    elif G.coords.shape[1] == 3:
        if not QtGui.QApplication.instance():
            QtGui.QApplication([])
        widget = gl.GLViewWidget()
        widget.opts['distance'] = 10
        widget.show()
        widget.setWindowTitle(plot_name)
        if show_edges:
            x, y, z = _get_coords(G)
            pos = np.stack((x, y, z), axis=1)
            g = gl.GLLinePlotItem(pos=pos, mode='lines', color=G.plotting['edge_color'])
            widget.addItem(g)
        gp = gl.GLScatterPlotItem(pos=G.coords, size=vertex_size / 3, color=G.plotting['vertex_color'])
        widget.addItem(gp)
        global _qtg_widgets
        _qtg_widgets.append(widget)