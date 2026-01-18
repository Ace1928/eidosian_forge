from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_common(verb, cmd):
    if verb:
        if verb.get('all'):
            cmd += ' {all}'.format(all='all')
        if verb.get('console'):
            cmd += ' {console}'.format(console='console')
        if verb.get('message_limit'):
            cmd += ' message-limit {message_limit}'.format(message_limit=verb['message_limit'])
        if verb.get('discriminator'):
            cmd += ' discriminator {discriminator}'.format(discriminator=verb.get('discriminator'))
        if verb.get('filtered'):
            cmd += ' {filtered}'.format(filtered='filtered')
        if verb.get('xml'):
            cmd += ' {xml}'.format(xml='xml')
        if verb.get('size'):
            cmd += ' {size}'.format(size=verb['size'])
        if verb.get('severity'):
            cmd += ' {severity}'.format(severity=verb['severity'])
        if verb.get('except_severity'):
            cmd += ' except {exceptSev}'.format(exceptSev=verb['except_severity'])
        if verb.get('tag'):
            cmd += ' {tag}'.format(tag=verb['tag'])
        if verb.get('text'):
            cmd += ' string {tag}'.format(tag=verb['text'])
        if verb.get('esm'):
            cmd += ' esm {tag}'.format(tag=verb['esm'])
        if verb.get('trap'):
            cmd += ' trap {tag}'.format(tag=verb['trap'])
    return cmd