from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class KeyPairModule(OpenStackModule):
    argument_spec = dict(name=dict(required=True), public_key=dict(), public_key_file=dict(), state=dict(default='present', choices=['absent', 'present', 'replace']))
    module_kwargs = dict(mutually_exclusive=[['public_key', 'public_key_file']])

    def _system_state_change(self, keypair):
        state = self.params['state']
        if state == 'present' and (not keypair):
            return True
        if state == 'absent' and keypair:
            return True
        return False

    def run(self):
        state = self.params['state']
        name = self.params['name']
        public_key = self.params['public_key']
        if self.params['public_key_file']:
            with open(self.params['public_key_file']) as public_key_fh:
                public_key = public_key_fh.read()
        keypair = self.conn.compute.find_keypair(name)
        if self.ansible.check_mode:
            self.exit_json(changed=self._system_state_change(keypair))
        changed = False
        if state in ('present', 'replace'):
            if keypair and keypair['name'] == name:
                if public_key and public_key != keypair['public_key']:
                    if state == 'present':
                        self.fail_json(msg='Key name %s present but key hash not the same as offered. Delete key first.' % name)
                    else:
                        self.conn.compute.delete_keypair(keypair)
                        keypair = self.conn.create_keypair(name, public_key)
                        changed = True
            else:
                keypair = self.conn.create_keypair(name, public_key)
                changed = True
            self.exit_json(changed=changed, keypair=keypair.to_dict(computed=False))
        elif state == 'absent':
            if keypair:
                self.conn.compute.delete_keypair(keypair)
                self.exit_json(changed=True)
            self.exit_json(changed=False)