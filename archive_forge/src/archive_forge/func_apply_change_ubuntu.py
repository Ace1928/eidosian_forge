from __future__ import absolute_import, division, print_function
import os
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.mh.deco import check_mode_skip
from ansible_collections.community.general.plugins.module_utils.locale_gen import locale_runner, locale_gen_runner
def apply_change_ubuntu(self, targetState, name):
    """Create or remove locale.

        Keyword arguments:
        targetState -- Desired state, either present or absent.
        name -- Name including encoding such as de_CH.UTF-8.
        """
    runner = locale_gen_runner(self.module)
    if targetState == 'present':
        with runner() as ctx:
            ctx.run()
    else:
        with open('/var/lib/locales/supported.d/local', 'r') as fr:
            content = fr.readlines()
        with open('/var/lib/locales/supported.d/local', 'w') as fw:
            for line in content:
                locale, charset = line.split(' ')
                if locale != name:
                    fw.write(line)
        with runner('purge') as ctx:
            ctx.run()