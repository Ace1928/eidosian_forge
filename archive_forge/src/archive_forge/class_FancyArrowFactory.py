import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
class FancyArrowFactory:
    """Draw arrows with `matplotlib.patches.FancyarrowPatch`"""

    class ConnectionStyleFactory:

        def __init__(self, connectionstyles, selfloop_height, ax=None):
            import matplotlib as mpl
            import matplotlib.path
            import numpy as np
            self.ax = ax
            self.mpl = mpl
            self.np = np
            self.base_connection_styles = [mpl.patches.ConnectionStyle(cs) for cs in connectionstyles]
            self.n = len(self.base_connection_styles)
            self.selfloop_height = selfloop_height

        def curved(self, edge_index):
            return self.base_connection_styles[edge_index % self.n]

        def self_loop(self, edge_index):

            def self_loop_connection(posA, posB, *args, **kwargs):
                if not self.np.all(posA == posB):
                    raise nx.NetworkXError('`self_loop` connection style methodis only to be used for self-loops')
                data_loc = self.ax.transData.inverted().transform(posA)
                v_shift = 0.1 * self.selfloop_height
                h_shift = v_shift * 0.5
                path = self.np.asarray([[0, v_shift], [h_shift, v_shift], [h_shift, 0], [0, 0], [-h_shift, 0], [-h_shift, v_shift], [0, v_shift]])
                if edge_index % 4:
                    x, y = path.T
                    for _ in range(edge_index % 4):
                        x, y = (y, -x)
                    path = self.np.array([x, y]).T
                return self.mpl.path.Path(self.ax.transData.transform(data_loc + path), [1, 4, 4, 4, 4, 4, 4])
            return self_loop_connection

    def __init__(self, edge_pos, edgelist, nodelist, edge_indices, node_size, selfloop_height, connectionstyle='arc3', node_shape='o', arrowstyle='-', arrowsize=10, edge_color='k', alpha=None, linewidth=1.0, style='solid', min_source_margin=0, min_target_margin=0, ax=None):
        import matplotlib as mpl
        import matplotlib.patches
        import matplotlib.pyplot as plt
        import numpy as np
        if isinstance(connectionstyle, str):
            connectionstyle = [connectionstyle]
        elif np.iterable(connectionstyle):
            connectionstyle = list(connectionstyle)
        else:
            msg = 'ConnectionStyleFactory arg `connectionstyle` must be str or iterable'
            raise nx.NetworkXError(msg)
        self.ax = ax
        self.mpl = mpl
        self.np = np
        self.edge_pos = edge_pos
        self.edgelist = edgelist
        self.nodelist = nodelist
        self.node_shape = node_shape
        self.min_source_margin = min_source_margin
        self.min_target_margin = min_target_margin
        self.edge_indices = edge_indices
        self.node_size = node_size
        self.connectionstyle_factory = self.ConnectionStyleFactory(connectionstyle, selfloop_height, ax)
        self.arrowstyle = arrowstyle
        self.arrowsize = arrowsize
        self.arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
        self.linewidth = linewidth
        self.style = style
        if isinstance(arrowsize, list) and len(arrowsize) != len(edge_pos):
            raise ValueError('arrowsize should have the same length as edgelist')

    def __call__(self, i):
        (x1, y1), (x2, y2) = self.edge_pos[i]
        shrink_source = 0
        shrink_target = 0
        if self.np.iterable(self.node_size):
            source, target = self.edgelist[i][:2]
            source_node_size = self.node_size[self.nodelist.index(source)]
            target_node_size = self.node_size[self.nodelist.index(target)]
            shrink_source = self.to_marker_edge(source_node_size, self.node_shape)
            shrink_target = self.to_marker_edge(target_node_size, self.node_shape)
        else:
            shrink_source = self.to_marker_edge(self.node_size, self.node_shape)
            shrink_target = shrink_source
        shrink_source = max(shrink_source, self.min_source_margin)
        shrink_target = max(shrink_target, self.min_target_margin)
        if isinstance(self.arrowsize, list):
            mutation_scale = self.arrowsize[i]
        else:
            mutation_scale = self.arrowsize
        if len(self.arrow_colors) > i:
            arrow_color = self.arrow_colors[i]
        elif len(self.arrow_colors) == 1:
            arrow_color = self.arrow_colors[0]
        else:
            arrow_color = self.arrow_colors[i % len(self.arrow_colors)]
        if self.np.iterable(self.linewidth):
            if len(self.linewidth) > i:
                linewidth = self.linewidth[i]
            else:
                linewidth = self.linewidth[i % len(self.linewidth)]
        else:
            linewidth = self.linewidth
        if self.np.iterable(self.style) and (not isinstance(self.style, str)) and (not isinstance(self.style, tuple)):
            if len(self.style) > i:
                linestyle = self.style[i]
            else:
                linestyle = self.style[i % len(self.style)]
        else:
            linestyle = self.style
        if x1 == x2 and y1 == y2:
            connectionstyle = self.connectionstyle_factory.self_loop(self.edge_indices[i])
        else:
            connectionstyle = self.connectionstyle_factory.curved(self.edge_indices[i])
        return self.mpl.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=self.arrowstyle, shrinkA=shrink_source, shrinkB=shrink_target, mutation_scale=mutation_scale, color=arrow_color, linewidth=linewidth, connectionstyle=connectionstyle, linestyle=linestyle, zorder=1)

    def to_marker_edge(self, marker_size, marker):
        if marker in 's^>v<d':
            return self.np.sqrt(2 * marker_size) / 2
        else:
            return self.np.sqrt(marker_size) / 2