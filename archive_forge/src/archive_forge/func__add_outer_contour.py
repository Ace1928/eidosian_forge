import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _add_outer_contour(all_contours, all_values, all_areas, all_colors, values, val_outer, v_min, v_max, colors, color_min, color_max):
    """
    Utility function for _contour_trace

    Adds the background color to fill gaps outside of computed contours.

    To compute the background color, the color of the contour with largest
    area (``val_outer``) is used. As background color, we choose the next
    color value in the direction of the extrema of the colormap.

    Then we add information for the outer contour for the different lists
    provided as arguments.

    A discrete colormap with all used colors is also returned (to be used
    by colorscale trace).
    """
    outer_contour = 20 * np.array([[0, 0, 1], [0, 1, 0.5]]).T
    all_contours = [outer_contour] + all_contours
    delta_values = np.diff(values)[0]
    values = np.concatenate(([values[0] - delta_values], values, [values[-1] + delta_values]))
    colors = np.concatenate(([color_min], colors, [color_max]))
    index = np.nonzero(values == val_outer)[0][0]
    if index < len(values) / 2:
        index -= 1
    else:
        index += 1
    all_colors = [colors[index]] + all_colors
    all_values = [values[index]] + all_values
    all_areas = [0] + all_areas
    used_colors = [color for color in colors if color in all_colors]
    color_number = len(used_colors)
    scale = np.linspace(0, 1, color_number + 1)
    discrete_cm = []
    for i, color in enumerate(used_colors):
        discrete_cm.append([scale[i], used_colors[i]])
        discrete_cm.append([scale[i + 1], used_colors[i]])
    discrete_cm.append([scale[color_number], used_colors[color_number - 1]])
    return (all_contours, all_values, all_areas, all_colors, discrete_cm)