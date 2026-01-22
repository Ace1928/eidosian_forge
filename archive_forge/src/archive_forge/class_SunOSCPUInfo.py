import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
class SunOSCPUInfo(CPUInfoBase):
    info = None

    def __init__(self):
        if self.info is not None:
            return
        info = command_info(arch='arch', mach='mach', uname_i='uname_i', isainfo_b='isainfo -b', isainfo_n='isainfo -n')
        info['uname_X'] = key_value_from_command('uname -X', sep='=')
        for line in command_by_line('psrinfo -v 0'):
            m = re.match('\\s*The (?P<p>[\\w\\d]+) processor operates at', line)
            if m:
                info['processor'] = m.group('p')
                break
        self.__class__.info = info

    def _not_impl(self):
        pass

    def _is_i386(self):
        return self.info['isainfo_n'] == 'i386'

    def _is_sparc(self):
        return self.info['isainfo_n'] == 'sparc'

    def _is_sparcv9(self):
        return self.info['isainfo_n'] == 'sparcv9'

    def _getNCPUs(self):
        return int(self.info['uname_X'].get('NumCPU', 1))

    def _is_sun4(self):
        return self.info['arch'] == 'sun4'

    def _is_SUNW(self):
        return re.match('SUNW', self.info['uname_i']) is not None

    def _is_sparcstation5(self):
        return re.match('.*SPARCstation-5', self.info['uname_i']) is not None

    def _is_ultra1(self):
        return re.match('.*Ultra-1', self.info['uname_i']) is not None

    def _is_ultra250(self):
        return re.match('.*Ultra-250', self.info['uname_i']) is not None

    def _is_ultra2(self):
        return re.match('.*Ultra-2', self.info['uname_i']) is not None

    def _is_ultra30(self):
        return re.match('.*Ultra-30', self.info['uname_i']) is not None

    def _is_ultra4(self):
        return re.match('.*Ultra-4', self.info['uname_i']) is not None

    def _is_ultra5_10(self):
        return re.match('.*Ultra-5_10', self.info['uname_i']) is not None

    def _is_ultra5(self):
        return re.match('.*Ultra-5', self.info['uname_i']) is not None

    def _is_ultra60(self):
        return re.match('.*Ultra-60', self.info['uname_i']) is not None

    def _is_ultra80(self):
        return re.match('.*Ultra-80', self.info['uname_i']) is not None

    def _is_ultraenterprice(self):
        return re.match('.*Ultra-Enterprise', self.info['uname_i']) is not None

    def _is_ultraenterprice10k(self):
        return re.match('.*Ultra-Enterprise-10000', self.info['uname_i']) is not None

    def _is_sunfire(self):
        return re.match('.*Sun-Fire', self.info['uname_i']) is not None

    def _is_ultra(self):
        return re.match('.*Ultra', self.info['uname_i']) is not None

    def _is_cpusparcv7(self):
        return self.info['processor'] == 'sparcv7'

    def _is_cpusparcv8(self):
        return self.info['processor'] == 'sparcv8'

    def _is_cpusparcv9(self):
        return self.info['processor'] == 'sparcv9'