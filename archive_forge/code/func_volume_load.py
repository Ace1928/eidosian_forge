from __future__ import absolute_import, division, print_function
import json  # noqa: E402
from ansible.module_utils.basic import AnsibleModule  # noqa: E402
def volume_load(module, executable):
    changed = True
    command = [executable, 'volume', 'import', module.params['volume'], module.params['src']]
    src = module.params['src']
    if module.check_mode:
        return (changed, '', '', '', command)
    rc, out, err = module.run_command(command)
    if rc != 0:
        module.fail_json(msg='Error importing volume %s: %s' % (src, err))
    rc, out2, err2 = module.run_command([executable, 'volume', 'inspect', module.params['volume']])
    if rc != 0:
        module.fail_json(msg='Volume %s inspection failed: %s' % (module.params['volume'], err2))
    try:
        info = json.loads(out2)[0]
    except Exception as e:
        module.fail_json(msg='Could not parse JSON from volume %s: %s' % (module.params['volume'], e))
    return (changed, out, err, info, command)