import logging
from os import getenv
from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker
def tsrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.slack.tqdm(range(*args), **kwargs)`."""
    return tqdm_slack(range(*args), **kwargs)