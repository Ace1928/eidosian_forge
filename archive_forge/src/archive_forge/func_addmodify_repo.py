from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves import configparser, StringIO
from io import open
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def addmodify_repo(module, repodata, old_repos, zypper_version, warnings):
    """Adds the repo, removes old repos before, that would conflict."""
    repo = repodata['url']
    cmd = _get_cmd(module, 'addrepo', '--check')
    if repodata['name']:
        cmd.extend(['--name', repodata['name']])
    if repodata['priority']:
        if zypper_version >= LooseVersion('1.12.25'):
            cmd.extend(['--priority', str(repodata['priority'])])
        else:
            warnings.append('Setting priority only available for zypper >= 1.12.25. Ignoring priority argument.')
    if repodata['enabled'] == '0':
        cmd.append('--disable')
    if zypper_version >= LooseVersion('1.6.2'):
        if repodata['gpgcheck'] == '1':
            cmd.append('--gpgcheck')
        else:
            cmd.append('--no-gpgcheck')
    else:
        warnings.append('Enabling/disabling gpgcheck only available for zypper >= 1.6.2. Using zypper default value.')
    if repodata['autorefresh'] == '1':
        cmd.append('--refresh')
    cmd.append(repo)
    if not repo.endswith('.repo'):
        cmd.append(repodata['alias'])
    if old_repos is not None:
        for oldrepo in old_repos:
            remove_repo(module, oldrepo['url'])
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    return (rc, stdout, stderr)