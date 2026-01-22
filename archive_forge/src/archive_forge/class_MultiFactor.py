from keystoneauth1.identity.v3 import base
from keystoneauth1 import loading
class MultiFactor(base.Auth):
    """A plugin for authenticating with multiple auth methods.

    :param string auth_url: Identity service endpoint for authentication.
    :param string auth_methods: names of the methods to authenticate with.
    :param string trust_id: Trust ID for trust scoping.
    :param string system_scope: System information to scope to.
    :param string domain_id: Domain ID for domain scoping.
    :param string domain_name: Domain name for domain scoping.
    :param string project_id: Project ID for project scoping.
    :param string project_name: Project name for project scoping.
    :param string project_domain_id: Project's domain ID for project.
    :param string project_domain_name: Project's domain name for project.
    :param bool reauthenticate: Allow fetching a new token if the current one
                                is going to expire. (optional) default True

    Also accepts various keyword args based on which methods are specified.
    """

    def __init__(self, auth_url, auth_methods, **kwargs):
        method_instances = []
        method_keys = set()
        for method in auth_methods:
            method_class = loading.get_plugin_loader(method).plugin_class._auth_method_class
            method_kwargs = {}
            for key in method_class._method_parameters:
                method_keys.add(key)
                method_kwargs[key] = kwargs.get(key, None)
            method_instances.append(method_class(**method_kwargs))
        for key in method_keys:
            kwargs.pop(key, None)
        super(MultiFactor, self).__init__(auth_url, method_instances, **kwargs)