import wandb
from wandb import util
from wandb.plots.utils import (

    ExplainText adds support for eli5's LIME based TextExplainer.
    Arguments:
     text (str): Text to explain
     probas (black-box classification pipeline): A function which
                    takes a list of strings (documents) and returns a matrix
                    of shape (n_samples, n_classes) with probability values,
                    i.e. a row per document and a column per output label.
    Returns:
     Nothing. To see plots, go to your W&B run page.
    Example:
     wandb.log({'roc': wandb.plots.ExplainText(text, probas)})
    