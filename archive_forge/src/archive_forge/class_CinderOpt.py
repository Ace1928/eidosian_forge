import os
from keystoneauth1 import loading
from keystoneauth1 import plugin
class CinderOpt(loading.Opt):

    @property
    def argparse_args(self):
        return ['--%s' % o.name for o in self._all_opts]

    @property
    def argparse_default(self):
        for o in self._all_opts:
            v = os.environ.get('Cinder_%s' % o.name.replace('-', '_').upper())
            if v:
                return v
        return self.default