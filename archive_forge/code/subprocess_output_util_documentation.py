import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
Handle SSH system failures coming from a subprocess.

    Args:
        e: The `ProcessRunnerException` to handle.
        first_conn_refused_time:
            The time (as reported by this function) or None,
            indicating the last time a CONN_REFUSED error was caught.

            After exceeding a patience value, the program will be aborted
            since SSH will likely never recover.
        retry_interval: The interval after which the command will be retried,
                        used here just to inform the user.
    