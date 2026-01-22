from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves import configparser, StringIO
from io import open
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _parse_repos(module):
    """parses the output of zypper --xmlout repos and return a parse repo dictionary"""
    cmd = _get_cmd(module, '--xmlout', 'repos')
    if not HAS_XML:
        module.fail_json(msg=missing_required_lib('python-xml'), exception=XML_IMP_ERR)
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    if rc == 0:
        repos = []
        dom = parseXML(stdout)
        repo_list = dom.getElementsByTagName('repo')
        for repo in repo_list:
            opts = {}
            for o in REPO_OPTS:
                opts[o] = repo.getAttribute(o)
            opts['url'] = repo.getElementsByTagName('url')[0].firstChild.data
            repos.append(opts)
        return repos
    elif rc == 6:
        return []
    else:
        module.fail_json(msg='Failed to execute "%s"' % ' '.join(cmd), rc=rc, stdout=stdout, stderr=stderr)