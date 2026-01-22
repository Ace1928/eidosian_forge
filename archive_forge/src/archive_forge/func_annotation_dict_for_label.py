from collections.abc import Sequence
from plotly import exceptions
from plotly.colors import (
def annotation_dict_for_label(text, lane, num_of_lanes, subplot_spacing, row_col='col', flipped=True, right_side=True, text_color='#0f0f0f'):
    """
    Returns annotation dict for label of n labels of a 1xn or nx1 subplot.

    :param (str) text: the text for a label.
    :param (int) lane: the label number for text. From 1 to n inclusive.
    :param (int) num_of_lanes: the number 'n' of rows or columns in subplot.
    :param (float) subplot_spacing: the value for the horizontal_spacing and
        vertical_spacing params in your plotly.tools.make_subplots() call.
    :param (str) row_col: choose whether labels are placed along rows or
        columns.
    :param (bool) flipped: flips text by 90 degrees. Text is printed
        horizontally if set to True and row_col='row', or if False and
        row_col='col'.
    :param (bool) right_side: only applicable if row_col is set to 'row'.
    :param (str) text_color: color of the text.
    """
    l = (1 - (num_of_lanes - 1) * subplot_spacing) / num_of_lanes
    if not flipped:
        xanchor = 'center'
        yanchor = 'middle'
        if row_col == 'col':
            x = (lane - 1) * (l + subplot_spacing) + 0.5 * l
            y = 1.03
            textangle = 0
        elif row_col == 'row':
            y = (lane - 1) * (l + subplot_spacing) + 0.5 * l
            x = 1.03
            textangle = 90
    elif row_col == 'col':
        xanchor = 'center'
        yanchor = 'bottom'
        x = (lane - 1) * (l + subplot_spacing) + 0.5 * l
        y = 1.0
        textangle = 270
    elif row_col == 'row':
        yanchor = 'middle'
        y = (lane - 1) * (l + subplot_spacing) + 0.5 * l
        if right_side:
            x = 1.0
            xanchor = 'left'
        else:
            x = -0.01
            xanchor = 'right'
        textangle = 0
    annotation_dict = dict(textangle=textangle, xanchor=xanchor, yanchor=yanchor, x=x, y=y, showarrow=False, xref='paper', yref='paper', text=text, font=dict(size=13, color=text_color))
    return annotation_dict