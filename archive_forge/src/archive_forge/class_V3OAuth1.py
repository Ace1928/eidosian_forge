from keystoneauth1.extras.oauth1 import v3
from keystoneauth1 import loading
class V3OAuth1(loading.BaseIdentityLoader):

    @property
    def plugin_class(self):
        return v3.OAuth1

    @property
    def available(self):
        return v3.oauth1 is not None

    def get_options(self):
        options = super(V3OAuth1, self).get_options()
        options.extend([loading.Opt('consumer-key', required=True, help='OAuth Consumer ID/Key'), loading.Opt('consumer-secret', required=True, help='OAuth Consumer Secret'), loading.Opt('access-key', required=True, help='OAuth Access Key'), loading.Opt('access-secret', required=True, help='OAuth Access Secret')])
        return options