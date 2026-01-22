import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _assert_effective_filters_if_needed(self):
    """Assert that useless filter combinations are avoided.

        In effective mode, the following filter combinations are useless, since
        they would always return an empty list of role assignments:
        - group id, since no group assignment is returned in effective mode;
        - domain id and inherited, since no domain inherited assignment is
        returned in effective mode.

        """
    if self._effective:
        if flask.request.args.get('group.id'):
            msg = _('Combining effective and group filter will always result in an empty list.')
            raise exception.ValidationError(msg)
        if self._inherited and flask.request.args.get('scope.domain.id'):
            msg = _('Combining effective, domain and inherited filters will always result in an empty list.')
            raise exception.ValidationError(msg)