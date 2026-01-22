import logging
import sys
import warnings
class _FileFormatter(logging.Formatter):
    """Customize the logging format for logging handler"""
    _LOG_MESSAGE_BEGIN = '%(asctime)s.%(msecs)03d %(process)d %(levelname)s %(name)s '
    _LOG_MESSAGE_CONTEXT = '[%(cloud)s %(username)s %(project)s] '
    _LOG_MESSAGE_END = '%(message)s'
    _LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, options=None, config=None, **kwargs):
        context = {}
        if options:
            context = {'cloud': getattr(options, 'cloud', ''), 'project': getattr(options, 'os_project_name', ''), 'username': getattr(options, 'username', '')}
        elif config:
            context = {'cloud': config.config.get('cloud', ''), 'project': config.auth.get('project_name', ''), 'username': config.auth.get('username', '')}
        if context:
            self.fmt = self._LOG_MESSAGE_BEGIN + self._LOG_MESSAGE_CONTEXT % context + self._LOG_MESSAGE_END
        else:
            self.fmt = self._LOG_MESSAGE_BEGIN + self._LOG_MESSAGE_END
        logging.Formatter.__init__(self, self.fmt, self._LOG_DATE_FORMAT)