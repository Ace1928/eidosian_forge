import logging
from time import monotonic as monotonic_clock
def uninstall_filter():
    """Uninstall the rate filter installed by install_filter().

    Do nothing if the filter was already uninstalled.
    """
    if install_filter.log_filter is None:
        return
    logging.setLoggerClass(install_filter.logger_class)
    for logger in _iter_loggers():
        logger.removeFilter(install_filter.log_filter)
    install_filter.logger_class = None
    install_filter.log_filter = None