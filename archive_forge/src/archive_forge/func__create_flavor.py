from novaclient.tests.functional import base
def _create_flavor(self, swap=None):
    flv_name = self.name_generate()
    cmd = 'flavor-create %s auto 512 1 1'
    if swap:
        cmd = cmd + ' --swap %s' % swap
    out = self.nova(cmd % flv_name)
    self.addCleanup(self.nova, 'flavor-delete %s' % flv_name)
    return (out, flv_name)