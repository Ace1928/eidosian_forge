from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
class PodmanNetwork:
    """Perform network tasks.

    Manages podman network, inspects it and checks its current state
    """

    def __init__(self, module, name):
        """Initialize PodmanNetwork class.

        Arguments:
            module {obj} -- ansible module object
            name {str} -- name of network
        """
        super(PodmanNetwork, self).__init__()
        self.module = module
        self.name = name
        self.stdout, self.stderr = ('', '')
        self.info = self.get_info()
        self.version = self._get_podman_version()
        self.diff = {}
        self.actions = []

    @property
    def exists(self):
        """Check if network exists."""
        return bool(self.info != {})

    @property
    def different(self):
        """Check if network is different."""
        diffcheck = PodmanNetworkDiff(self.module, self.info, self.version)
        is_different = diffcheck.is_different()
        diffs = diffcheck.diff
        if self.module._diff and is_different and diffs['before'] and diffs['after']:
            self.diff['before'] = '\n'.join(['%s - %s' % (k, v) for k, v in sorted(diffs['before'].items())]) + '\n'
            self.diff['after'] = '\n'.join(['%s - %s' % (k, v) for k, v in sorted(diffs['after'].items())]) + '\n'
        return is_different

    def get_info(self):
        """Inspect network and gather info about it."""
        rc, out, err = self.module.run_command([self.module.params['executable'], b'network', b'inspect', self.name])
        return json.loads(out)[0] if rc == 0 else {}

    def _get_podman_version(self):
        rc, out, err = self.module.run_command([self.module.params['executable'], b'--version'])
        if rc != 0 or not out or 'version' not in out:
            self.module.fail_json(msg='%s run failed!' % self.module.params['executable'])
        return out.split('version')[1].strip()

    def _perform_action(self, action):
        """Perform action with network.

        Arguments:
            action {str} -- action to perform - create, stop, delete
        """
        b_command = PodmanNetworkModuleParams(action, self.module.params, self.version, self.module).construct_command_from_params()
        full_cmd = ' '.join([self.module.params['executable'], 'network'] + [to_native(i) for i in b_command])
        self.module.log('PODMAN-NETWORK-DEBUG: %s' % full_cmd)
        self.actions.append(full_cmd)
        if not self.module.check_mode:
            rc, out, err = self.module.run_command([self.module.params['executable'], b'network'] + b_command, expand_user_and_vars=False)
            self.stdout = out
            self.stderr = err
            if rc != 0:
                self.module.fail_json(msg="Can't %s network %s" % (action, self.name), stdout=out, stderr=err)

    def delete(self):
        """Delete the network."""
        self._perform_action('delete')

    def create(self):
        """Create the network."""
        self._perform_action('create')

    def recreate(self):
        """Recreate the network."""
        self.delete()
        self.create()