import configparser
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_utils import units
from oslo_log._i18n import _
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
def _setup_logging_from_conf(conf, project, version):
    log_root = getLogger(None).logger
    for handler in list(log_root.handlers):
        log_root.removeHandler(handler)
    logpath = _get_log_file_path(conf)
    if logpath:
        if conf.watch_log_file and platform.system() == 'Linux':
            from oslo_log import watchers
            file_handler = watchers.FastWatchedFileHandler
            filelog = file_handler(logpath)
        elif conf.log_rotation_type.lower() == 'interval':
            file_handler = logging.handlers.TimedRotatingFileHandler
            when = conf.log_rotate_interval_type.lower()
            interval_type = LOG_ROTATE_INTERVAL_MAPPING[when]
            if interval_type == 'w':
                interval_type = interval_type + str(conf.log_rotate_interval)
            filelog = file_handler(logpath, when=interval_type, interval=conf.log_rotate_interval, backupCount=conf.max_logfile_count)
        elif conf.log_rotation_type.lower() == 'size':
            file_handler = logging.handlers.RotatingFileHandler
            maxBytes = conf.max_logfile_size_mb * units.Mi
            filelog = file_handler(logpath, maxBytes=maxBytes, backupCount=conf.max_logfile_count)
        else:
            file_handler = logging.handlers.WatchedFileHandler
            filelog = file_handler(logpath)
        log_root.addHandler(filelog)
    if conf.use_stderr:
        streamlog = handlers.ColorHandler()
        log_root.addHandler(streamlog)
    if conf.use_journal:
        if syslog is None:
            raise RuntimeError('syslog is not available on this platform')
        facility = _find_facility(conf.syslog_log_facility)
        journal = handlers.OSJournalHandler(facility=facility)
        log_root.addHandler(journal)
    if conf.use_eventlog:
        if platform.system() == 'Windows':
            eventlog = logging.handlers.NTEventLogHandler(project)
            log_root.addHandler(eventlog)
        else:
            raise RuntimeError(_('Windows Event Log is not available on this platform.'))
    if not logpath and (not conf.use_stderr) and (not conf.use_journal):
        streamlog = handlers.ColorHandler(sys.stdout)
        log_root.addHandler(streamlog)
    if conf.publish_errors:
        handler = importutils.import_object('oslo_messaging.notify.log_handler.PublishErrorsHandler', logging.ERROR)
        log_root.addHandler(handler)
    if conf.use_syslog:
        if syslog is None:
            raise RuntimeError('syslog is not available on this platform')
        facility = _find_facility(conf.syslog_log_facility)
        syslog_handler = handlers.OSSysLogHandler(facility=facility)
        log_root.addHandler(syslog_handler)
    datefmt = conf.log_date_format
    if not conf.use_json:
        for handler in log_root.handlers:
            handler.setFormatter(formatters.ContextFormatter(project=project, version=version, datefmt=datefmt, config=conf))
    else:
        for handler in log_root.handlers:
            handler.setFormatter(formatters.JSONFormatter(datefmt=datefmt))
    _refresh_root_level(conf.debug)
    for pair in conf.default_log_levels:
        mod, _sep, level_name = pair.partition('=')
        logger = logging.getLogger(mod)
        numeric_level = None
        try:
            numeric_level = int(level_name)
        except ValueError:
            pass
        if numeric_level is not None:
            logger.setLevel(numeric_level)
        else:
            logger.setLevel(level_name)
    if conf.rate_limit_burst >= 1 and conf.rate_limit_interval >= 1:
        from oslo_log import rate_limit
        rate_limit.install_filter(conf.rate_limit_burst, conf.rate_limit_interval, conf.rate_limit_except)