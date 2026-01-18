import wandb
from wandb import util
from wandb.plots.utils import (

    Generates a heatmap.

    Arguments:
     matrix_values (arr): 2D dataset of shape x_labels * y_labels, containing
                        heatmap values that can be coerced into an ndarray.
     x_labels  (list): Named labels for rows (x_axis).
     y_labels  (list): Named labels for columns (y_axis).
     show_text (bool): Show text values in heatmap cells.

    Returns:
     Nothing. To see plots, go to your W&B run page then expand the 'media' tab
           under 'auto visualizations'.

    Example:
     wandb.log({'heatmap': wandb.plots.HeatMap(x_labels, y_labels,
                matrix_values)})
    