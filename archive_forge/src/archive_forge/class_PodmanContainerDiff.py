from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
class PodmanContainerDiff:

    def __init__(self, module, module_params, info, image_info, podman_version):
        self.module = module
        self.module_params = module_params
        self.version = podman_version
        self.default_dict = None
        self.info = lower_keys(info)
        self.image_info = lower_keys(image_info)
        self.params = self.defaultize()
        self.diff = {'before': {}, 'after': {}}
        self.non_idempotent = {}

    def defaultize(self):
        params_with_defaults = {}
        self.default_dict = PodmanDefaults(self.image_info, self.version).default_dict()
        for p in self.module_params:
            if self.module_params[p] is None and p in self.default_dict:
                params_with_defaults[p] = self.default_dict[p]
            else:
                params_with_defaults[p] = self.module_params[p]
        return params_with_defaults

    def _createcommand(self, argument):
        """Returns list of values for given argument from CreateCommand
        from Podman container inspect output.

        Args:
            argument (str): argument name

        Returns:

            all_values: list of values for given argument from createcommand
        """
        if 'createcommand' not in self.info['config']:
            return []
        cr_com = self.info['config']['createcommand']
        argument_values = ARGUMENTS_OPTS_DICT.get(argument, [argument])
        all_values = []
        for arg in argument_values:
            for ind, cr_opt in enumerate(cr_com):
                if arg == cr_opt:
                    if not cr_com[ind + 1].startswith('-'):
                        all_values.append(cr_com[ind + 1])
                    else:
                        return [True]
                if cr_opt.startswith('%s=' % arg):
                    all_values.append(cr_opt.split('=', 1)[1])
        return all_values

    def _diff_update_and_compare(self, param_name, before, after):
        if before != after:
            self.diff['before'].update({param_name: before})
            self.diff['after'].update({param_name: after})
            return True
        return False

    def diffparam_annotation(self):
        before = self.info['config']['annotations'] or {}
        after = before.copy()
        if self.module_params['annotation'] is not None:
            after.update(self.params['annotation'])
        return self._diff_update_and_compare('annotation', before, after)

    def diffparam_env_host(self):
        before = False
        after = self.params['env_host']
        return self._diff_update_and_compare('env_host', before, after)

    def diffparam_blkio_weight(self):
        before = self.info['hostconfig']['blkioweight']
        after = self.params['blkio_weight']
        return self._diff_update_and_compare('blkio_weight', before, after)

    def diffparam_blkio_weight_device(self):
        before = self.info['hostconfig']['blkioweightdevice']
        if before == [] and self.module_params['blkio_weight_device'] is None:
            after = []
        else:
            after = self.params['blkio_weight_device']
        return self._diff_update_and_compare('blkio_weight_device', before, after)

    def diffparam_cap_add(self):
        before = self.info['effectivecaps'] or []
        before = [i.lower() for i in before]
        after = []
        if self.module_params['cap_add'] is not None:
            for cap in self.module_params['cap_add']:
                cap = cap.lower()
                cap = cap if cap.startswith('cap_') else 'cap_' + cap
                after.append(cap)
        after += before
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('cap_add', before, after)

    def diffparam_cap_drop(self):
        before = self.info['effectivecaps'] or []
        before = [i.lower() for i in before]
        after = before[:]
        if self.module_params['cap_drop'] is not None:
            for cap in self.module_params['cap_drop']:
                cap = cap.lower()
                cap = cap if cap.startswith('cap_') else 'cap_' + cap
                if cap in after:
                    after.remove(cap)
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('cap_drop', before, after)

    def diffparam_cgroup_parent(self):
        before = self.info['hostconfig']['cgroupparent']
        after = self.params['cgroup_parent']
        if after is None:
            after = before
        return self._diff_update_and_compare('cgroup_parent', before, after)

    def diffparam_cgroups(self):
        if 'cgroups' in self.info['hostconfig']:
            before = self.info['hostconfig']['cgroups']
            after = self.params['cgroups']
            return self._diff_update_and_compare('cgroups', before, after)
        return False

    def diffparam_cidfile(self):
        before = self.info['hostconfig']['containeridfile']
        after = self.params['cidfile']
        labels = self.info['config']['labels'] or {}
        if 'podman_systemd_unit' in labels:
            after = before
        return self._diff_update_and_compare('cidfile', before, after)

    def diffparam_command(self):
        if self.module_params['command'] is not None:
            before = self.info['config']['cmd']
            after = self.params['command']
            if isinstance(after, str):
                after = shlex.split(after)
            return self._diff_update_and_compare('command', before, after)
        return False

    def diffparam_conmon_pidfile(self):
        before = self.info['conmonpidfile']
        if self.module_params['conmon_pidfile'] is None:
            after = before
        else:
            after = self.params['conmon_pidfile']
        return self._diff_update_and_compare('conmon_pidfile', before, after)

    def diffparam_cpu_period(self):
        before = self.info['hostconfig']['cpuperiod']
        after = self.params['cpu_period'] or before
        return self._diff_update_and_compare('cpu_period', before, after)

    def diffparam_cpu_quota(self):
        before = self.info['hostconfig']['cpuquota']
        after = self.params['cpu_quota'] or before
        return self._diff_update_and_compare('cpu_quota', before, after)

    def diffparam_cpu_rt_period(self):
        before = self.info['hostconfig']['cpurealtimeperiod']
        after = self.params['cpu_rt_period']
        return self._diff_update_and_compare('cpu_rt_period', before, after)

    def diffparam_cpu_rt_runtime(self):
        before = self.info['hostconfig']['cpurealtimeruntime']
        after = self.params['cpu_rt_runtime']
        return self._diff_update_and_compare('cpu_rt_runtime', before, after)

    def diffparam_cpu_shares(self):
        before = self.info['hostconfig']['cpushares']
        after = self.params['cpu_shares']
        return self._diff_update_and_compare('cpu_shares', before, after)

    def diffparam_cpus(self):
        before = self.info['hostconfig']['nanocpus'] / 1000000000
        after = float(self.params['cpus'] or before)
        return self._diff_update_and_compare('cpus', before, after)

    def diffparam_cpuset_cpus(self):
        before = self.info['hostconfig']['cpusetcpus']
        after = self.params['cpuset_cpus']
        return self._diff_update_and_compare('cpuset_cpus', before, after)

    def diffparam_cpuset_mems(self):
        before = self.info['hostconfig']['cpusetmems']
        after = self.params['cpuset_mems']
        return self._diff_update_and_compare('cpuset_mems', before, after)

    def diffparam_device(self):
        before = [':'.join([i['pathonhost'], i['pathincontainer']]) for i in self.info['hostconfig']['devices']]
        if not before and 'createcommand' in self.info['config']:
            before = [i.lower() for i in self._createcommand('--device')]
        before = [':'.join((i, i)) if len(i.split(':')) == 1 else i for i in before]
        after = [':'.join(i.split(':')[:2]) for i in self.params['device']]
        after = [':'.join((i, i)) if len(i.split(':')) == 1 else i for i in after]
        before, after = ([i.lower() for i in before], [i.lower() for i in after])
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('devices', before, after)

    def diffparam_device_read_bps(self):
        before = self.info['hostconfig']['blkiodevicereadbps'] or []
        before = ['%s:%s' % (i['path'], i['rate']) for i in before]
        after = self.params['device_read_bps'] or []
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('device_read_bps', before, after)

    def diffparam_device_read_iops(self):
        before = self.info['hostconfig']['blkiodevicereadiops'] or []
        before = ['%s:%s' % (i['path'], i['rate']) for i in before]
        after = self.params['device_read_iops'] or []
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('device_read_iops', before, after)

    def diffparam_device_write_bps(self):
        before = self.info['hostconfig']['blkiodevicewritebps'] or []
        before = ['%s:%s' % (i['path'], i['rate']) for i in before]
        after = self.params['device_write_bps'] or []
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('device_write_bps', before, after)

    def diffparam_device_write_iops(self):
        before = self.info['hostconfig']['blkiodevicewriteiops'] or []
        before = ['%s:%s' % (i['path'], i['rate']) for i in before]
        after = self.params['device_write_iops'] or []
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('device_write_iops', before, after)

    def diffparam_env(self):
        env_before = self.info['config']['env'] or {}
        before = {i.split('=')[0]: '='.join(i.split('=')[1:]) for i in env_before}
        after = before.copy()
        if self.params['env']:
            after.update({k: str(v) for k, v in self.params['env'].items()})
        return self._diff_update_and_compare('env', before, after)

    def diffparam_etc_hosts(self):
        if self.info['hostconfig']['extrahosts']:
            before = dict([i.split(':', 1) for i in self.info['hostconfig']['extrahosts']])
        else:
            before = {}
        after = self.params['etc_hosts']
        return self._diff_update_and_compare('etc_hosts', before, after)

    def diffparam_group_add(self):
        before = self.info['hostconfig']['groupadd']
        after = self.params['group_add']
        return self._diff_update_and_compare('group_add', before, after)

    def diffparam_healthcheck(self):
        before = ''
        if 'healthcheck' in self.info['config']:
            if len(self.info['config']['healthcheck']['test']) > 1:
                before = self.info['config']['healthcheck']['test'][1]
        after = self.params['healthcheck'] or before
        return self._diff_update_and_compare('healthcheck', before, after)

    def diffparam_healthcheck_failure_action(self):
        if 'healthcheckonfailureaction' in self.info['config']:
            before = self.info['config']['healthcheckonfailureaction']
        else:
            before = ''
        after = self.params['healthcheck_failure_action'] or before
        return self._diff_update_and_compare('healthcheckonfailureaction', before, after)

    def diffparam_hostname(self):
        before = self.info['config']['hostname']
        after = self.params['hostname'] or before
        return self._diff_update_and_compare('hostname', before, after)

    def diffparam_image(self):
        before_id = self.info['image'] or self.info['rootfs']
        after_id = self.image_info['id']
        if before_id == after_id:
            return self._diff_update_and_compare('image', before_id, after_id)
        is_rootfs = self.info['rootfs'] != '' or self.params['rootfs']
        before = self.info['config']['image'] or before_id
        after = self.params['image']
        mode = self.params['image_strict'] or is_rootfs
        if mode is None or not mode:
            before = before.replace(':latest', '')
            after = after.replace(':latest', '')
            before = before.split('/')[-1]
            after = after.split('/')[-1]
        else:
            return self._diff_update_and_compare('image', before_id, after_id)
        return self._diff_update_and_compare('image', before, after)

    def diffparam_ipc(self):
        before = self.info['hostconfig']['ipcmode']
        after = self.params['ipc']
        if self.params['pod'] and (not self.module_params['ipc']):
            after = before
        return self._diff_update_and_compare('ipc', before, after)

    def diffparam_label(self):
        before = self.info['config']['labels'] or {}
        after = self.image_info.get('labels') or {}
        if self.params['label']:
            after.update({str(k).lower(): str(v) for k, v in self.params['label'].items()})
        if 'podman_systemd_unit' in before:
            after.pop('podman_systemd_unit', None)
            before.pop('podman_systemd_unit', None)
        return self._diff_update_and_compare('label', before, after)

    def diffparam_log_driver(self):
        before = self.info['hostconfig']['logconfig']['type']
        if self.module_params['log_driver'] is not None:
            after = self.params['log_driver']
        else:
            after = before
        return self._diff_update_and_compare('log_driver', before, after)

    def diffparam_log_opt(self):
        before, after = ({}, {})
        path_before = None
        if 'logpath' in self.info:
            path_before = self.info['logpath']
        if 'logconfig' in self.info['hostconfig'] and 'path' in self.info['hostconfig']['logconfig']:
            path_before = self.info['hostconfig']['logconfig']['path']
        if path_before is not None:
            if self.module_params['log_opt'] and 'path' in self.module_params['log_opt'] and (self.module_params['log_opt']['path'] is not None):
                path_after = self.params['log_opt']['path']
            else:
                path_after = path_before
            if path_before != path_after:
                before.update({'log-path': path_before})
                after.update({'log-path': path_after})
        tag_before = None
        if 'logtag' in self.info:
            tag_before = self.info['logtag']
        if 'logconfig' in self.info['hostconfig'] and 'tag' in self.info['hostconfig']['logconfig']:
            tag_before = self.info['hostconfig']['logconfig']['tag']
        if tag_before is not None:
            if self.module_params['log_opt'] and 'tag' in self.module_params['log_opt'] and (self.module_params['log_opt']['tag'] is not None):
                tag_after = self.params['log_opt']['tag']
            else:
                tag_after = ''
            if tag_before != tag_after:
                before.update({'log-tag': tag_before})
                after.update({'log-tag': tag_after})
        return self._diff_update_and_compare('log_opt', before, after)

    def diffparam_mac_address(self):
        before = str(self.info['networksettings']['macaddress'])
        if not before and self.info['networksettings'].get('networks'):
            nets = self.info['networksettings']['networks']
            macs = [nets[i]['macaddress'] for i in nets if nets[i]['macaddress']]
            if macs:
                before = macs[0]
        if not before and 'createcommand' in self.info['config']:
            before = [i.lower() for i in self._createcommand('--mac-address')]
            before = before[0] if before else ''
        if self.module_params['mac_address'] is not None:
            after = self.params['mac_address']
        else:
            after = before
        return self._diff_update_and_compare('mac_address', before, after)

    def diffparam_network(self):
        net_mode_before = self.info['hostconfig']['networkmode']
        net_mode_after = ''
        before = list(self.info['networksettings'].get('networks', {}))
        if before == ['podman']:
            before = []
        if net_mode_before == 'slirp4netns' and 'createcommand' in self.info['config']:
            cr_net = [i.lower() for i in self._createcommand('--network')]
            for cr_net_opt in cr_net:
                if 'slirp4netns:' in cr_net_opt:
                    before = [cr_net_opt]
        after = self.params['network'] or []
        if not self.module_params['network'] and self.params['pod']:
            after = before
            return self._diff_update_and_compare('network', before, after)
        if after in [['bridge'], ['host'], ['slirp4netns'], ['none']]:
            net_mode_after = after[0]
        if net_mode_after and (not before):
            net_mode_after = net_mode_after.replace('bridge', 'default')
            net_mode_after = net_mode_after.replace('slirp4netns', 'default')
            net_mode_before = net_mode_before.replace('bridge', 'default')
            net_mode_before = net_mode_before.replace('slirp4netns', 'default')
            return self._diff_update_and_compare('network', net_mode_before, net_mode_after)
        if 'container' in net_mode_before:
            for netw in after:
                if 'container' in netw:
                    before = after = netw
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('network', before, after)

    def diffparam_oom_score_adj(self):
        before = self.info['hostconfig']['oomscoreadj']
        after = self.params['oom_score_adj']
        return self._diff_update_and_compare('oom_score_adj', before, after)

    def diffparam_privileged(self):
        before = self.info['hostconfig']['privileged']
        after = self.params['privileged']
        return self._diff_update_and_compare('privileged', before, after)

    def diffparam_pid(self):

        def get_container_id_by_name(name):
            rc, podman_inspect_info, err = self.module.run_command([self.module.params['executable'], 'inspect', name, '-f', '{{.Id}}'])
            if rc != 0:
                return None
            return podman_inspect_info.strip()
        before = self.info['hostconfig']['pidmode']
        after = self.params['pid']
        if after is not None and 'container:' in after and ('container:' in before):
            if after.split(':')[1] == before.split(':')[1]:
                return self._diff_update_and_compare('pid', before, after)
            after = 'container:' + get_container_id_by_name(after.split(':')[1])
        return self._diff_update_and_compare('pid', before, after)

    def diffparam_publish(self):

        def compose(p, h):
            s = ':'.join([str(h['hostport']), p.replace('/tcp', '')]).strip(':')
            if h['hostip']:
                return ':'.join([h['hostip'], s])
            return s
        ports = self.info['hostconfig']['portbindings']
        before = []
        for port, hosts in ports.items():
            if hosts:
                for h in hosts:
                    before.append(compose(port, h))
        after = self.params['publish'] or []
        if self.params['publish_all']:
            image_ports = self.image_info.get('config', {}).get('exposedports', {})
            if image_ports:
                after += list(image_ports.keys())
        after = [i.replace('/tcp', '').replace('[', '').replace(']', '').replace('0.0.0.0:', '') for i in after]
        for ports in after:
            if '-' in ports:
                return self._diff_update_and_compare('publish', '', '')
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('publish', before, after)

    def diffparam_read_only(self):
        before = self.info['hostconfig']['readonlyrootfs']
        after = self.params['read_only']
        return self._diff_update_and_compare('read_only', before, after)

    def diffparam_restart_policy(self):
        before = self.info['hostconfig']['restartpolicy']['name']
        before_max_count = int(self.info['hostconfig']['restartpolicy'].get('maximumretrycount', 0))
        after = self.params['restart_policy'] or ''
        if ':' in after:
            after, after_max_count = after.rsplit(':', 1)
            after_max_count = int(after_max_count)
        else:
            after_max_count = 0
        before = '%s:%i' % (before, before_max_count)
        after = '%s:%i' % (after, after_max_count)
        return self._diff_update_and_compare('restart_policy', before, after)

    def diffparam_rm(self):
        before = self.info['hostconfig']['autoremove']
        after = self.params['rm']
        return self._diff_update_and_compare('rm', before, after)

    def diffparam_security_opt(self):
        unsorted_before = self.info['hostconfig']['securityopt']
        unsorted_after = self.params['security_opt']
        before = sorted((item for element in unsorted_before for item in element.split(',') if 'apparmor=container-default' not in item))
        after = sorted(list(set(unsorted_after)))
        return self._diff_update_and_compare('security_opt', before, after)

    def diffparam_stop_signal(self):
        before = normalize_signal(self.info['config']['stopsignal'])
        after = normalize_signal(self.params['stop_signal'])
        return self._diff_update_and_compare('stop_signal', before, after)

    def diffparam_timezone(self):
        before = self.info['config'].get('timezone')
        after = self.params['timezone']
        return self._diff_update_and_compare('timezone', before, after)

    def diffparam_tty(self):
        before = self.info['config']['tty']
        after = self.params['tty']
        return self._diff_update_and_compare('tty', before, after)

    def diffparam_user(self):
        before = self.info['config']['user']
        after = self.params['user']
        return self._diff_update_and_compare('user', before, after)

    def diffparam_ulimit(self):
        after = self.params['ulimit'] or []
        if 'createcommand' in self.info['config']:
            before = self._createcommand('--ulimit')
            before, after = (sorted(before), sorted(after))
            return self._diff_update_and_compare('ulimit', before, after)
        if after:
            ulimits = self.info['hostconfig']['ulimits']
            before = {u['name'].replace('rlimit_', ''): '%s:%s' % (u['soft'], u['hard']) for u in ulimits}
            after = {i.split('=')[0]: i.split('=')[1] for i in self.params['ulimit']}
            new_before = []
            new_after = []
            for u in list(after.keys()):
                if u in before and '-1' not in after[u]:
                    new_before.append([u, before[u]])
                    new_after.append([u, after[u]])
            return self._diff_update_and_compare('ulimit', new_before, new_after)
        return self._diff_update_and_compare('ulimit', '', '')

    def diffparam_uts(self):
        before = self.info['hostconfig']['utsmode']
        after = self.params['uts']
        if self.params['pod'] and (not self.module_params['uts']):
            after = before
        return self._diff_update_and_compare('uts', before, after)

    def diffparam_volume(self):

        def clean_volume(x):
            """Remove trailing and double slashes from volumes."""
            if not x.rstrip('/'):
                return '/'
            return x.replace('//', '/').rstrip('/')
        before = self.info['mounts']
        before_local_vols = []
        if before:
            volumes = []
            local_vols = []
            for m in before:
                if m['type'] != 'volume':
                    volumes.append([clean_volume(m['source']), clean_volume(m['destination'])])
                elif m['type'] == 'volume':
                    local_vols.append([m['name'], clean_volume(m['destination'])])
            before = [':'.join(v) for v in volumes]
            before_local_vols = [':'.join(v) for v in local_vols]
        if self.params['volume'] is not None:
            after = [':'.join([clean_volume(i) for i in v.split(':')[:2]]) for v in self.params['volume']]
        else:
            after = []
        if before_local_vols:
            after = list(set(after).difference(before_local_vols))
        before, after = (sorted(list(set(before))), sorted(list(set(after))))
        return self._diff_update_and_compare('volume', before, after)

    def diffparam_volumes_from(self):
        before = self.info['hostconfig'].get('volumesfrom', []) or []
        after = self.params['volumes_from'] or []
        return self._diff_update_and_compare('volumes_from', before, after)

    def diffparam_workdir(self):
        before = self.info['config']['workingdir']
        after = self.params['workdir']
        return self._diff_update_and_compare('workdir', before, after)

    def is_different(self):
        diff_func_list = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith('diffparam')]
        fail_fast = not bool(self.module._diff)
        different = False
        for func_name in diff_func_list:
            dff_func = getattr(self, func_name)
            if dff_func():
                if fail_fast:
                    return True
                different = True
        for p in self.non_idempotent:
            if self.module_params[p] is not None and self.module_params[p] not in [{}, [], '']:
                different = True
        return different