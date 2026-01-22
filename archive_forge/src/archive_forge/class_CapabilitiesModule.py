from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class CapabilitiesModule(object):
    platform = 'Linux'
    distribution = None

    def __init__(self, module):
        self.module = module
        self.path = module.params['path'].strip()
        self.capability = module.params['capability'].strip().lower()
        self.state = module.params['state']
        self.getcap_cmd = module.get_bin_path('getcap', required=True)
        self.setcap_cmd = module.get_bin_path('setcap', required=True)
        self.capability_tup = self._parse_cap(self.capability, op_required=self.state == 'present')
        self.run()

    def run(self):
        current = self.getcap(self.path)
        caps = [cap[0] for cap in current]
        if self.state == 'present' and self.capability_tup not in current:
            if self.module.check_mode:
                self.module.exit_json(changed=True, msg='capabilities changed')
            else:
                current = list(filter(lambda x: x[0] != self.capability_tup[0], current))
                current.append(self.capability_tup)
                self.module.exit_json(changed=True, state=self.state, msg='capabilities changed', stdout=self.setcap(self.path, current))
        elif self.state == 'absent' and self.capability_tup[0] in caps:
            if self.module.check_mode:
                self.module.exit_json(changed=True, msg='capabilities changed')
            else:
                current = filter(lambda x: x[0] != self.capability_tup[0], current)
                self.module.exit_json(changed=True, state=self.state, msg='capabilities changed', stdout=self.setcap(self.path, current))
        self.module.exit_json(changed=False, state=self.state)

    def getcap(self, path):
        rval = []
        cmd = '%s -v %s' % (self.getcap_cmd, path)
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0 or stderr != '':
            self.module.fail_json(msg='Unable to get capabilities of %s' % path, stdout=stdout.strip(), stderr=stderr)
        if stdout.strip() != path:
            if ' =' in stdout:
                caps = stdout.split(' =')[1].strip().split()
            else:
                caps = stdout.split()[1].strip().split()
            for cap in caps:
                cap = cap.lower()
                if ',' in cap:
                    cap_group = cap.split(',')
                    cap_group[-1], op, flags = self._parse_cap(cap_group[-1])
                    for subcap in cap_group:
                        rval.append((subcap, op, flags))
                else:
                    rval.append(self._parse_cap(cap))
        return rval

    def setcap(self, path, caps):
        caps = ' '.join([''.join(cap) for cap in caps])
        cmd = "%s '%s' %s" % (self.setcap_cmd, caps, path)
        rc, stdout, stderr = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Unable to set capabilities of %s' % path, stdout=stdout, stderr=stderr)
        else:
            return stdout

    def _parse_cap(self, cap, op_required=True):
        opind = -1
        try:
            i = 0
            while opind == -1:
                opind = cap.find(OPS[i])
                i += 1
        except Exception:
            if op_required:
                self.module.fail_json(msg="Couldn't find operator (one of: %s)" % str(OPS))
            else:
                return (cap, None, None)
        op = cap[opind]
        cap, flags = cap.split(op)
        return (cap, op, flags)