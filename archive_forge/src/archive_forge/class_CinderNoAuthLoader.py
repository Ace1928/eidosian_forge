import os
from keystoneauth1 import loading
from keystoneauth1 import plugin
class CinderNoAuthLoader(loading.BaseLoader):
    plugin_class = CinderNoAuthPlugin

    def get_options(self):
        options = super(CinderNoAuthLoader, self).get_options()
        options.extend([CinderOpt('user-id', help='User ID', required=True, metavar='<cinder user id>'), CinderOpt('project-id', help='Project ID', metavar='<cinder project id>'), CinderOpt('endpoint', help='Cinder endpoint', dest='endpoint', required=True, metavar='<cinder endpoint>')])
        return options