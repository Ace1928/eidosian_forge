import logging
import os
from oslo_middleware.healthcheck import opts
from oslo_middleware.healthcheck import pluginbase
class DisableByFileHealthcheck(pluginbase.HealthcheckBaseExtension):
    """DisableByFile healthcheck middleware plugin

    This plugin checks presence of a file to report if the service
    is unavailable or not.

    Example of middleware configuration:

    .. code-block:: ini

      [filter:healthcheck]
      paste.filter_factory = oslo_middleware:Healthcheck.factory
      path = /healthcheck
      backends = disable_by_file
      disable_by_file_path = /var/run/nova/healthcheck_disable
      # set to True to enable detailed output, False is the default
      detailed = False
    """

    def __init__(self, *args, **kwargs):
        super(DisableByFileHealthcheck, self).__init__(*args, **kwargs)
        self.oslo_conf.register_opts(opts.DISABLE_BY_FILE_OPTS, group='healthcheck')

    def healthcheck(self, server_port):
        path = self._conf_get('disable_by_file_path')
        if not path:
            LOG.warning('DisableByFile healthcheck middleware enabled without disable_by_file_path set')
            return pluginbase.HealthcheckResult(available=True, reason='OK', details="No 'disable_by_file_path' configuration value specified")
        elif not os.path.exists(path):
            return pluginbase.HealthcheckResult(available=True, reason='OK', details="Path '%s' was not found" % path)
        else:
            return pluginbase.HealthcheckResult(available=False, reason='DISABLED BY FILE', details="Path '%s' was found" % path)