import re
from hacking import core
@core.flake8ext
def accepted_log_levels(logical_line, filename):
    """In Sahara we use only 5 log levels.

    This check is needed because we don't want new contributors to
    use deprecated log levels.
    S374
    """
    ignore_dirs = ['sahara/db/templates', 'sahara/tests']
    for directory in ignore_dirs:
        if directory in filename:
            return
    msg = 'S374 You used deprecated log level. Accepted log levels are %(levels)s' % {'levels': ALL_LOG_LEVELS}
    if logical_line.startswith('LOG.'):
        if not RE_ACCEPTED_LOG_LEVELS.search(logical_line):
            yield (0, msg)