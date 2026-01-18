import typing as t
from collections.abc import Iterable
import wandb
Construct a line series plot.

    Arguments:
        xs (array of arrays, or array): Array of arrays of x values
        ys (array of arrays): Array of y values
        keys (array): Array of labels for the line plots
        title (string): Plot title.
        xname: Title of x-axis
        split_table (bool): If True, adds "Custom Chart Tables/" to the key of the table so that it's logged in a different section.

    Returns:
        A plot object, to be passed to wandb.log()

    Example:
        When logging a singular array for xs, all ys are plotted against that xs
        <!--yeadoc-test:plot-line-series-single-->
        ```python
        import wandb

        run = wandb.init()
        xs = [i for i in range(10)]
        ys = [[i for i in range(10)], [i**2 for i in range(10)]]
        run.log(
            {"line-series-plot1": wandb.plot.line_series(xs, ys, title="title", xname="step")}
        )
        run.finish()
        ```
        xs can also contain an array of arrays for having different steps for each metric
        <!--yeadoc-test:plot-line-series-double-->
        ```python
        import wandb

        run = wandb.init()
        xs = [[i for i in range(10)], [2 * i for i in range(10)]]
        ys = [[i for i in range(10)], [i**2 for i in range(10)]]
        run.log(
            {"line-series-plot2": wandb.plot.line_series(xs, ys, title="title", xname="step")}
        )
        run.finish()
        ```
    