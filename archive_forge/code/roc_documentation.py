import wandb
from wandb import util
from wandb.plots.utils import (

    Calculates receiver operating characteristic scores and visualizes them as the
     ROC curve.

    Arguments:
     y_true (arr): Test set labels.
     y_probas (arr): Test set predicted probabilities.
     labels (list): Named labels for target varible (y). Makes plots easier to
                     read by replacing target values with corresponding index.
                     For example labels= ['dog', 'cat', 'owl'] all 0s are
                     replaced by 'dog', 1s by 'cat'.

    Returns:
     Nothing. To see plots, go to your W&B run page then expand the 'media' tab
           under 'auto visualizations'.

    Example:
     wandb.log({'roc': wandb.plots.ROC(y_true, y_probas, labels)})
    