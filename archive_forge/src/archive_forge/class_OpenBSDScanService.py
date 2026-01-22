from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
class OpenBSDScanService(BaseService):

    def query_rcctl(self, cmd):
        svcs = []
        rc, stdout, stderr = self.module.run_command('%s ls %s' % (self.rcctl_path, cmd))
        if 'needs root privileges' in stderr.lower():
            self.module.warn('rcctl requires root privileges')
        else:
            for svc in stdout.split('\n'):
                if svc == '':
                    continue
                else:
                    svcs.append(svc)
        return svcs

    def get_info(self, name):
        info = {}
        rc, stdout, stderr = self.module.run_command('%s get %s' % (self.rcctl_path, name))
        if 'needs root privileges' in stderr.lower():
            self.module.warn('rcctl requires root privileges')
        else:
            undy = '%s_' % name
            for variable in stdout.split('\n'):
                if variable == '' or '=' not in variable:
                    continue
                else:
                    k, v = variable.replace(undy, '', 1).split('=')
                    info[k] = v
        return info

    def gather_services(self):
        services = {}
        self.rcctl_path = self.module.get_bin_path('rcctl')
        if self.rcctl_path:
            for svc in self.query_rcctl('all'):
                services[svc] = {'name': svc, 'source': 'rcctl', 'rogue': False}
                services[svc].update(self.get_info(svc))
            for svc in self.query_rcctl('on'):
                services[svc].update({'status': 'enabled'})
            for svc in self.query_rcctl('started'):
                services[svc].update({'state': 'running'})
            for svc in self.query_rcctl('failed'):
                services[svc].update({'state': 'failed'})
            for svc in services.keys():
                if services[svc].get('status') is None:
                    services[svc].update({'status': 'disabled'})
                if services[svc].get('state') is None:
                    services[svc].update({'state': 'stopped'})
            for svc in self.query_rcctl('rogue'):
                services[svc]['rogue'] = True
        return services