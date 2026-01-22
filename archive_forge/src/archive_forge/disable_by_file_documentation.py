import logging
import os
from oslo_middleware.healthcheck import opts
from oslo_middleware.healthcheck import pluginbase
DisableByFile healthcheck middleware plugin

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
    