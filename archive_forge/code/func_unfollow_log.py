from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def unfollow_log(module, le_path, logs):
    """ Unfollows one or more logs if followed. """
    removed_count = 0
    for log in logs:
        if not query_log_status(module, le_path, log):
            continue
        if module.check_mode:
            module.exit_json(changed=True)
        rc, out, err = module.run_command([le_path, 'rm', log])
        if query_log_status(module, le_path, log):
            module.fail_json(msg="failed to remove '%s': %s" % (log, err.strip()))
        removed_count += 1
    if removed_count > 0:
        module.exit_json(changed=True, msg='removed %d package(s)' % removed_count)
    module.exit_json(changed=False, msg='logs(s) already unfollowed')