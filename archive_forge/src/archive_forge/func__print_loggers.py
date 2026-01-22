import logging
import re
from logging.config import dictConfig
import threading
from typing import Union
def _print_loggers():
    """Print a formatted list of loggers and their handlers for debugging."""
    loggers = {logging.root.name: logging.root}
    loggers.update(dict(sorted(logging.root.manager.loggerDict.items())))
    for name, logger in loggers.items():
        if isinstance(logger, logging.Logger):
            print(f'  {name}: disabled={logger.disabled}, propagate={logger.propagate}')
            for handler in logger.handlers:
                print(f'    {handler}')