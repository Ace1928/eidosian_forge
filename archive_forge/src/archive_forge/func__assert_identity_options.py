from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
def _assert_identity_options(options):
    if options.get('username') and (not (options.get('user_domain_name') or options.get('user_domain_id'))):
        m = 'You have provided a username. In the V3 identity API a username is only unique within a domain so you must also provide either a user_domain_id or user_domain_name.'
        raise exceptions.OptionError(m)