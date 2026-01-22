import collections
def _build_subplot_title_annotations(subplot_titles, list_of_domains, title_edge='top', offset=0):
    x_dom = list_of_domains[::2]
    y_dom = list_of_domains[1::2]
    subtitle_pos_x = []
    subtitle_pos_y = []
    if title_edge == 'top':
        text_angle = 0
        xanchor = 'center'
        yanchor = 'bottom'
        for x_domains in x_dom:
            subtitle_pos_x.append(sum(x_domains) / 2.0)
        for y_domains in y_dom:
            subtitle_pos_y.append(y_domains[1])
        yshift = offset
        xshift = 0
    elif title_edge == 'bottom':
        text_angle = 0
        xanchor = 'center'
        yanchor = 'top'
        for x_domains in x_dom:
            subtitle_pos_x.append(sum(x_domains) / 2.0)
        for y_domains in y_dom:
            subtitle_pos_y.append(y_domains[0])
        yshift = -offset
        xshift = 0
    elif title_edge == 'right':
        text_angle = 90
        xanchor = 'left'
        yanchor = 'middle'
        for x_domains in x_dom:
            subtitle_pos_x.append(x_domains[1])
        for y_domains in y_dom:
            subtitle_pos_y.append(sum(y_domains) / 2.0)
        yshift = 0
        xshift = offset
    elif title_edge == 'left':
        text_angle = -90
        xanchor = 'right'
        yanchor = 'middle'
        for x_domains in x_dom:
            subtitle_pos_x.append(x_domains[0])
        for y_domains in y_dom:
            subtitle_pos_y.append(sum(y_domains) / 2.0)
        yshift = 0
        xshift = -offset
    else:
        raise ValueError("Invalid annotation edge '{edge}'".format(edge=title_edge))
    plot_titles = []
    for index in range(len(subplot_titles)):
        if not subplot_titles[index] or index >= len(subtitle_pos_y):
            pass
        else:
            annot = {'y': subtitle_pos_y[index], 'xref': 'paper', 'x': subtitle_pos_x[index], 'yref': 'paper', 'text': subplot_titles[index], 'showarrow': False, 'font': dict(size=16), 'xanchor': xanchor, 'yanchor': yanchor}
            if xshift != 0:
                annot['xshift'] = xshift
            if yshift != 0:
                annot['yshift'] = yshift
            if text_angle != 0:
                annot['textangle'] = text_angle
            plot_titles.append(annot)
    return plot_titles