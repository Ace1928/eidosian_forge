import functools
import sys
import flask
import oslo_i18n
from oslo_log import log
from oslo_middleware import healthcheck
import keystone.api
from keystone import exception
from keystone.oauth2 import handlers as oauth2_handlers
from keystone.receipt import handlers as receipt_handlers
from keystone.server.flask import common as ks_flask
from keystone.server.flask.request_processing import json_body
from keystone.server.flask.request_processing import req_logging
@fail_gracefully
def application_factory(name='public'):
    if name not in ('admin', 'public'):
        raise RuntimeError('Application name (for base_url lookup) must be either `admin` or `public`.')
    app = flask.Flask(name)
    for exc in exception.KEYSTONE_API_EXCEPTIONS:
        app.register_error_handler(exc, _handle_keystone_exception)
    app.register_error_handler(TypeError, _handle_unknown_keystone_exception)
    app.before_request(req_logging.log_request_info)
    app.before_request(json_body.json_body_before_request)
    app.after_request(_add_vary_x_auth_token_header)
    app.config.update(PROPAGATE_EXCEPTIONS=True)
    for api in keystone.api.__apis__:
        for api_bp in api.APIs:
            api_bp.instantiate_and_register_to_app(app)
    hc_app = healthcheck.Healthcheck.app_factory({}, oslo_config_project='keystone')
    app.wsgi_app = wsgi_dispatcher.DispatcherMiddleware(app.wsgi_app, {'/healthcheck': hc_app})
    return app