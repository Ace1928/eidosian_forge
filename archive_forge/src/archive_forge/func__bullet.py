import math
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
import plotly
import plotly.graph_objs as go
def _bullet(df, markers, measures, ranges, subtitles, titles, orientation, range_colors, measure_colors, horizontal_spacing, vertical_spacing, scatter_options, layout_options):
    num_of_lanes = len(df)
    num_of_rows = num_of_lanes if orientation == 'h' else 1
    num_of_cols = 1 if orientation == 'h' else num_of_lanes
    if not horizontal_spacing:
        horizontal_spacing = 1.0 / num_of_lanes
    if not vertical_spacing:
        vertical_spacing = 1.0 / num_of_lanes
    fig = plotly.subplots.make_subplots(num_of_rows, num_of_cols, print_grid=False, horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing)
    fig['layout'].update(dict(shapes=[]), title='Bullet Chart', height=600, width=1000, showlegend=False, barmode='stack', annotations=[], margin=dict(l=120 if orientation == 'h' else 80))
    fig['layout'].update(layout_options)
    if orientation == 'h':
        width_axis = 'yaxis'
        length_axis = 'xaxis'
    else:
        width_axis = 'xaxis'
        length_axis = 'yaxis'
    for key in fig['layout']:
        if 'xaxis' in key or 'yaxis' in key:
            fig['layout'][key]['showgrid'] = False
            fig['layout'][key]['zeroline'] = False
        if length_axis in key:
            fig['layout'][key]['tickwidth'] = 1
        if width_axis in key:
            fig['layout'][key]['showticklabels'] = False
            fig['layout'][key]['range'] = [0, 1]
    if num_of_lanes <= 1:
        fig['layout'][width_axis + '1']['domain'] = [0.4, 0.6]
    if not range_colors:
        range_colors = ['rgb(200, 200, 200)', 'rgb(245, 245, 245)']
    if not measure_colors:
        measure_colors = ['rgb(31, 119, 180)', 'rgb(176, 196, 221)']
    for row in range(num_of_lanes):
        for idx in range(len(df.iloc[row]['ranges'])):
            inter_colors = clrs.n_colors(range_colors[0], range_colors[1], len(df.iloc[row]['ranges']), 'rgb')
            x = [sorted(df.iloc[row]['ranges'])[-1 - idx]] if orientation == 'h' else [0]
            y = [0] if orientation == 'h' else [sorted(df.iloc[row]['ranges'])[-1 - idx]]
            bar = go.Bar(x=x, y=y, marker=dict(color=inter_colors[-1 - idx]), name='ranges', hoverinfo='x' if orientation == 'h' else 'y', orientation=orientation, width=2, base=0, xaxis='x{}'.format(row + 1), yaxis='y{}'.format(row + 1))
            fig.add_trace(bar)
        for idx in range(len(df.iloc[row]['measures'])):
            inter_colors = clrs.n_colors(measure_colors[0], measure_colors[1], len(df.iloc[row]['measures']), 'rgb')
            x = [sorted(df.iloc[row]['measures'])[-1 - idx]] if orientation == 'h' else [0.5]
            y = [0.5] if orientation == 'h' else [sorted(df.iloc[row]['measures'])[-1 - idx]]
            bar = go.Bar(x=x, y=y, marker=dict(color=inter_colors[-1 - idx]), name='measures', hoverinfo='x' if orientation == 'h' else 'y', orientation=orientation, width=0.4, base=0, xaxis='x{}'.format(row + 1), yaxis='y{}'.format(row + 1))
            fig.add_trace(bar)
        x = df.iloc[row]['markers'] if orientation == 'h' else [0.5]
        y = [0.5] if orientation == 'h' else df.iloc[row]['markers']
        markers = go.Scatter(x=x, y=y, name='markers', hoverinfo='x' if orientation == 'h' else 'y', xaxis='x{}'.format(row + 1), yaxis='y{}'.format(row + 1), **scatter_options)
        fig.add_trace(markers)
        title = df.iloc[row]['titles']
        if 'subtitles' in df:
            subtitle = '<br>{}'.format(df.iloc[row]['subtitles'])
        else:
            subtitle = ''
        label = '<b>{}</b>'.format(title) + subtitle
        annot = utils.annotation_dict_for_label(label, num_of_lanes - row if orientation == 'h' else row + 1, num_of_lanes, vertical_spacing if orientation == 'h' else horizontal_spacing, 'row' if orientation == 'h' else 'col', True if orientation == 'h' else False, False)
        fig['layout']['annotations'] += (annot,)
    return fig