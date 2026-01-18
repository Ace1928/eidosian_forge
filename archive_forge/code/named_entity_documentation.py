import wandb
from wandb import util
from wandb.plots.utils import (

    Adds support for spaCy's entity visualizer, which highlights named
        entities and their labels in a text.

    Arguments:
     docs (list, Doc, Span): Document(s) to visualize.

    Returns:
     Nothing. To see plots, go to your W&B run page.

    Example:
     wandb.log({'NER': wandb.plots.NER(docs=doc)})
    