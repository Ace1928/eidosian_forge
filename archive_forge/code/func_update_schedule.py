from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_schedule(module, array, snap_time, repl_time):
    """Update Protection Group Schedule"""
    changed = False
    try:
        schedule = array.get_pgroup(module.params['name'], schedule=True)
        retention = array.get_pgroup(module.params['name'], retention=True)
        if not schedule['replicate_blackout']:
            schedule['replicate_blackout'] = [{'start': 0, 'end': 0}]
    except Exception:
        module.fail_json(msg='Failed to get current schedule for pgroup {0}.'.format(module.params['name']))
    current_repl = {'replicate_frequency': schedule['replicate_frequency'], 'replicate_enabled': schedule['replicate_enabled'], 'target_days': retention['target_days'], 'replicate_at': schedule['replicate_at'], 'target_per_day': retention['target_per_day'], 'target_all_for': retention['target_all_for'], 'blackout_start': schedule['replicate_blackout'][0]['start'], 'blackout_end': schedule['replicate_blackout'][0]['end']}
    current_snap = {'days': retention['days'], 'snap_frequency': schedule['snap_frequency'], 'snap_enabled': schedule['snap_enabled'], 'snap_at': schedule['snap_at'], 'per_day': retention['per_day'], 'all_for': retention['all_for']}
    if module.params['schedule'] == 'snapshot':
        if not module.params['snap_frequency']:
            snap_frequency = current_snap['snap_frequency']
        elif not 300 <= module.params['snap_frequency'] <= 34560000:
            module.fail_json(msg='Snap Frequency support is out of range (300 to 34560000)')
        else:
            snap_frequency = module.params['snap_frequency']
        if module.params['enabled'] is None:
            snap_enabled = current_snap['snap_enabled']
        else:
            snap_enabled = module.params['enabled']
        if not module.params['snap_at']:
            snap_at = current_snap['snap_at']
        else:
            snap_at = _convert_to_minutes(module.params['snap_at'].upper())
        if not module.params['days']:
            if isinstance(module.params['days'], int):
                days = module.params['days']
            else:
                days = current_snap['days']
        elif module.params['days'] > 4000:
            module.fail_json(msg='Maximum value for days is 4000')
        else:
            days = module.params['days']
        if module.params['per_day'] is None:
            per_day = current_snap['per_day']
        elif module.params['per_day'] > 1440:
            module.fail_json(msg='Maximum value for per_day is 1440')
        else:
            per_day = module.params['per_day']
        if not module.params['all_for']:
            all_for = current_snap['all_for']
        elif module.params['all_for'] > 34560000:
            module.fail_json(msg='Maximum all_for value is 34560000')
        else:
            all_for = module.params['all_for']
        new_snap = {'days': days, 'snap_frequency': snap_frequency, 'snap_enabled': snap_enabled, 'snap_at': snap_at, 'per_day': per_day, 'all_for': all_for}
        module.warn('current {0}; new: {1}'.format(current_snap, new_snap))
        if current_snap != new_snap:
            changed = True
            if not module.check_mode:
                try:
                    array.set_pgroup(module.params['name'], snap_enabled=module.params['enabled'])
                    if snap_time:
                        array.set_pgroup(module.params['name'], snap_frequency=snap_frequency, snap_at=snap_at)
                    else:
                        array.set_pgroup(module.params['name'], snap_frequency=snap_frequency)
                    array.set_pgroup(module.params['name'], days=days, per_day=per_day, all_for=all_for)
                except Exception:
                    module.fail_json(msg='Failed to change snapshot schedule for pgroup {0}.'.format(module.params['name']))
    else:
        if not module.params['replicate_frequency']:
            replicate_frequency = current_repl['replicate_frequency']
        else:
            model = array.get(controllers=True)[0]['model']
            if '405' in model or '10' in model or 'CBS' in model:
                if not 900 <= module.params['replicate_frequency'] <= 34560000:
                    module.fail_json(msg='Replication Frequency support is out of range (900 to 34560000)')
                else:
                    replicate_frequency = module.params['replicate_frequency']
            elif not 300 <= module.params['replicate_frequency'] <= 34560000:
                module.fail_json(msg='Replication Frequency support is out of range (300 to 34560000)')
            else:
                replicate_frequency = module.params['replicate_frequency']
        if module.params['enabled'] is None:
            replicate_enabled = current_repl['replicate_enabled']
        else:
            replicate_enabled = module.params['enabled']
        if not module.params['replicate_at']:
            replicate_at = current_repl['replicate_at']
        else:
            replicate_at = _convert_to_minutes(module.params['replicate_at'].upper())
        if not module.params['target_days']:
            if isinstance(module.params['target_days'], int):
                target_days = module.params['target_days']
            else:
                target_days = current_repl['target_days']
        elif module.params['target_days'] > 4000:
            module.fail_json(msg='Maximum value for target_days is 4000')
        else:
            target_days = module.params['target_days']
        if not module.params['target_per_day']:
            if isinstance(module.params['target_per_day'], int):
                target_per_day = module.params['target_per_day']
            else:
                target_per_day = current_repl['target_per_day']
        elif module.params['target_per_day'] > 1440:
            module.fail_json(msg='Maximum value for target_per_day is 1440')
        else:
            target_per_day = module.params['target_per_day']
        if not module.params['target_all_for']:
            target_all_for = current_repl['target_all_for']
        elif module.params['target_all_for'] > 34560000:
            module.fail_json(msg='Maximum target_all_for value is 34560000')
        else:
            target_all_for = module.params['target_all_for']
        if not module.params['blackout_end']:
            blackout_end = current_repl['blackout_start']
        else:
            blackout_end = _convert_to_minutes(module.params['blackout_end'].upper())
        if not module.params['blackout_start']:
            blackout_start = current_repl['blackout_start']
        else:
            blackout_start = _convert_to_minutes(module.params['blackout_start'].upper())
        new_repl = {'replicate_frequency': replicate_frequency, 'replicate_enabled': replicate_enabled, 'target_days': target_days, 'replicate_at': replicate_at, 'target_per_day': target_per_day, 'target_all_for': target_all_for, 'blackout_start': blackout_start, 'blackout_end': blackout_end}
        if current_repl != new_repl:
            changed = True
            if not module.check_mode:
                blackout = {'start': blackout_start, 'end': blackout_end}
                try:
                    array.set_pgroup(module.params['name'], replicate_enabled=module.params['enabled'])
                    if repl_time:
                        array.set_pgroup(module.params['name'], replicate_frequency=replicate_frequency, replicate_at=replicate_at)
                    else:
                        array.set_pgroup(module.params['name'], replicate_frequency=replicate_frequency)
                    if blackout_start == 0:
                        array.set_pgroup(module.params['name'], replicate_blackout=None)
                    else:
                        array.set_pgroup(module.params['name'], replicate_blackout=blackout)
                    array.set_pgroup(module.params['name'], target_days=target_days, target_per_day=target_per_day, target_all_for=target_all_for)
                except Exception:
                    module.fail_json(msg='Failed to change replication schedule for pgroup {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)