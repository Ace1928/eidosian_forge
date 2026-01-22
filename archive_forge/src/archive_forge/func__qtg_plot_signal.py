from __future__ import division
import numpy as np
from pygsp import utils
def _qtg_plot_signal(G, signal, show_edges, plot_name, vertex_size, limits):
    qtg, gl, QtGui = _import_qtg()
    if G.coords.shape[1] == 2:
        window = qtg.GraphicsWindow(plot_name)
        view = window.addViewBox()
    elif G.coords.shape[1] == 3:
        if not QtGui.QApplication.instance():
            QtGui.QApplication([])
        widget = gl.GLViewWidget()
        widget.opts['distance'] = 10
        widget.show()
        widget.setWindowTitle(plot_name)
    if show_edges:
        if G.is_directed():
            raise NotImplementedError
        elif G.coords.shape[1] == 2:
            adj = _get_coords(G, edge_list=True)
            pen = tuple(np.array(G.plotting['edge_color']) * 255)
            g = qtg.GraphItem(pos=G.coords, adj=adj, symbolBrush=None, symbolPen=None, pen=pen)
            view.addItem(g)
        elif G.coords.shape[1] == 3:
            x, y, z = _get_coords(G)
            pos = np.stack((x, y, z), axis=1)
            g = gl.GLLinePlotItem(pos=pos, mode='lines', color=G.plotting['edge_color'])
            widget.addItem(g)
    pos = [1, 8, 24, 40, 56, 64]
    color = np.array([[0, 0, 143, 255], [0, 0, 255, 255], [0, 255, 255, 255], [255, 255, 0, 255], [255, 0, 0, 255], [128, 0, 0, 255]])
    cmap = qtg.ColorMap(pos, color)
    signal = 1 + 63 * (signal - limits[0]) / limits[1] - limits[0]
    if G.coords.shape[1] == 2:
        gp = qtg.ScatterPlotItem(G.coords[:, 0], G.coords[:, 1], size=vertex_size / 10, brush=cmap.map(signal, 'qcolor'))
        view.addItem(gp)
    if G.coords.shape[1] == 3:
        gp = gl.GLScatterPlotItem(pos=G.coords, size=vertex_size / 3, color=cmap.map(signal, 'float'))
        widget.addItem(gp)
    if G.coords.shape[1] == 2:
        global _qtg_windows
        _qtg_windows.append(window)
    elif G.coords.shape[1] == 3:
        global _qtg_widgets
        _qtg_widgets.append(widget)