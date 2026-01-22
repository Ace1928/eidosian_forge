from statsmodels.compat.python import lrange
def annotate_axes(index, labels, points, offset_points, size, ax, **kwargs):
    """
    Annotate Axes with labels, points, offset_points according to the
    given index.
    """
    for i in index:
        label = labels[i]
        point = points[i]
        offset = offset_points[i]
        ax.annotate(label, point, xytext=offset, textcoords='offset points', size=size, **kwargs)
    return ax