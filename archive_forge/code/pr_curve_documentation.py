from typing import Optional
import wandb
from wandb import util
from wandb.plots.utils import test_missing, test_types
Compute the tradeoff between precision and recall for different thresholds.

    A high area under the curve represents both high recall and high precision, where
    high precision relates to a low false positive rate, and high recall relates to a
    low false negative rate. High scores for both show that the classifier is returning
    accurate results (high precision), and returning a majority of all positive results
    (high recall). PR curve is useful when the classes are very imbalanced.

    Arguments:
        y_true (arr): true sparse labels y_probas (arr): Target scores, can either be
            probability estimates, confidence values, or non-thresholded measure of
            decisions. shape: (*y_true.shape, num_classes)
        labels (list): Named labels for target variable (y). Makes plots easier to read
            by replacing target values with corresponding index. For example labels =
            ['dog', 'cat', 'owl'] all 0s are replaced by 'dog', 1s by 'cat'.
        classes_to_plot (list): unique values of y_true to include in the plot
        interp_size (int): the recall values will be fixed to `interp_size` points
            uniform on [0, 1] and the precision will be interpolated for these recall
            values.
        split_table (bool): If True, adds "Custom Chart Tables/" to the key of the table so that it's logged in a different section.

    Returns:
        Nothing. To see plots, go to your W&B run page then expand the 'media' tab under
        'auto visualizations'.

    Example:
        ```
        wandb.log({"pr-curve": wandb.plot.pr_curve(y_true, y_probas, labels)})
        ```
    