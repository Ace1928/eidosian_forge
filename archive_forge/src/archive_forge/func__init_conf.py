import copy
import logging
import debtcollector
from oslo_config import cfg
from oslo_middleware import base
import webob.exc
def _init_conf(self):
    """Initialize this middleware from an oslo.config instance."""
    self.oslo_conf.register_opts(CORS_OPTS, 'cors')
    allowed_origin = self._conf_get('allowed_origin', 'cors')
    allow_credentials = self._conf_get('allow_credentials', 'cors')
    expose_headers = self._conf_get('expose_headers', 'cors')
    max_age = self._conf_get('max_age', 'cors')
    allow_methods = self._conf_get('allow_methods', 'cors')
    allow_headers = self._conf_get('allow_headers', 'cors')
    subgroup_opts = copy.deepcopy(CORS_OPTS)
    cfg.set_defaults(subgroup_opts, allow_credentials=allow_credentials, expose_headers=expose_headers, max_age=max_age, allow_methods=allow_methods, allow_headers=allow_headers)
    self.add_origin(allowed_origin=allowed_origin, allow_credentials=allow_credentials, expose_headers=expose_headers, max_age=max_age, allow_methods=allow_methods, allow_headers=allow_headers)
    for section in self.oslo_conf.list_all_sections():
        if section.startswith('cors.'):
            debtcollector.deprecate('Multiple configuration blocks are deprecated and will be removed in future versions. Please consolidate your configuration in the [cors] configuration block.')
            self.oslo_conf.register_opts(subgroup_opts, section)
            self.add_origin(**self.oslo_conf[section])