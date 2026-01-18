from celery import bootsteps
from celery.utils.log import get_logger
from .events import Events
Bootstep syncing state with neighbor workers.

    At startup, or upon consumer restart, this will:

    - Sync logical clocks.
    - Sync revoked tasks.

    