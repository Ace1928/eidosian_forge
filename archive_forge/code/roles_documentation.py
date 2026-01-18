import flask
import flask_restful
import http.client
from keystone.api._shared import implied_roles as shared
from keystone.assignment import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.server import flask as ks_flask
Delete implied role.

        DELETE /v3/roles/{prior_role_id}/implies/{implied_role_id}
        