import json
import os
import warnings
from . import _catboost
def write_plot_file(plot_file_stream):
    plot_file_stream.write('\n'.join(('<html>', '<head>', '<meta charset="utf-8" />', '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>', '<title>{}</title>'.format(plot_name), '</head>', '<body>')))
    for fig in figs:
        graph_div = plotly_plot(fig, output_type='div', show_link=False, include_plotlyjs=False)
        plot_file_stream.write('\n{}\n'.format(graph_div))
    plot_file_stream.write('</body>\n</html>')