from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_persistent(config_data):
    cmd = 'logging persistent'
    verb = config_data.get('persistent')
    if verb.get('url'):
        cmd += ' url {url}'.format(url=verb['url'])
    if verb.get('size'):
        cmd += ' size {size}'.format(size=verb['size'])
    if verb.get('filesize'):
        cmd += ' filesize {filesize}'.format(filesize=verb['filesize'])
    if verb.get('batch'):
        cmd += ' batch {batch}'.format(batch=verb['batch'])
    if verb.get('threshold'):
        cmd += ' threshold {threshold}'.format(threshold=verb['threshold'])
    if verb.get('immediate'):
        cmd += ' {immediate}'.format(immediate='immediate')
    if verb.get('protected'):
        cmd += ' {protected}'.format(protected='protected')
    if verb.get('notify'):
        cmd += ' {notify}'.format(notify='notify')
    return cmd