import os
from keystoneauth1 import loading
from keystoneauth1 import plugin
class AodhNoAuthLoader(loading.BaseLoader):
    plugin_class = AodhNoAuthPlugin

    def get_options(self):
        options = super(AodhNoAuthLoader, self).get_options()
        options.extend([AodhOpt('user-id', help='User ID', required=True), AodhOpt('project-id', help='Project ID', required=True), AodhOpt('roles', help='Roles', default='admin'), AodhOpt('aodh-endpoint', help='Aodh endpoint', dest='endpoint', required=True)])
        return options