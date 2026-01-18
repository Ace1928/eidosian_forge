import warnings
from billiard.common import TERM_SIGNAME
from kombu.matcher import match
from kombu.pidbox import Mailbox
from kombu.utils.compat import register_after_fork
from kombu.utils.functional import lazy
from kombu.utils.objects import cached_property
from celery.exceptions import DuplicateNodenameWarning
from celery.utils.log import get_logger
from celery.utils.text import pluralize
def time_limit(self, task_name, soft=None, hard=None, destination=None, **kwargs):
    """Tell workers to set time limits for a task by type.

        Arguments:
            task_name (str): Name of task to change time limits for.
            soft (float): New soft time limit (in seconds).
            hard (float): New hard time limit (in seconds).
            **kwargs (Any): arguments passed on to :meth:`broadcast`.
        """
    return self.broadcast('time_limit', arguments={'task_name': task_name, 'hard': hard, 'soft': soft}, destination=destination, **kwargs)