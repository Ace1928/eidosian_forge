import logging
import os
import re
import sys
import warnings
from datetime import timezone
from tzlocal import utils
def _get_localzone_name(_root='/'):
    """Tries to find the local timezone configuration.

    This method finds the timezone name, if it can, or it returns None.

    The parameter _root makes the function look for files like /etc/localtime
    beneath the _root directory. This is primarily used by the tests.
    In normal usage you call the function without parameters."""
    tzenv = utils._tz_name_from_env()
    if tzenv:
        return tzenv
    if os.path.exists(os.path.join(_root, 'system/bin/getprop')):
        log.debug('This looks like Termux')
        import subprocess
        try:
            androidtz = subprocess.check_output(['getprop', 'persist.sys.timezone']).strip().decode()
            return androidtz
        except (OSError, subprocess.CalledProcessError):
            log.debug("It's not termux?")
            pass
    found_configs = {}
    for configfile in ('etc/timezone', 'var/db/zoneinfo'):
        tzpath = os.path.join(_root, configfile)
        try:
            with open(tzpath) as tzfile:
                data = tzfile.read()
                log.debug(f'{tzpath} found, contents:\n {data}')
                etctz = data.strip('/ \t\r\n')
                if not etctz:
                    continue
                for etctz in etctz.splitlines():
                    if ' ' in etctz:
                        etctz, dummy = etctz.split(' ', 1)
                    if '#' in etctz:
                        etctz, dummy = etctz.split('#', 1)
                    if not etctz:
                        continue
                    found_configs[tzpath] = etctz.replace(' ', '_')
        except (OSError, UnicodeDecodeError):
            continue
    zone_re = re.compile('\\s*ZONE\\s*=\\s*\\"')
    timezone_re = re.compile('\\s*TIMEZONE\\s*=\\s*\\"')
    end_re = re.compile('"')
    for filename in ('etc/sysconfig/clock', 'etc/conf.d/clock'):
        tzpath = os.path.join(_root, filename)
        try:
            with open(tzpath, 'rt') as tzfile:
                data = tzfile.readlines()
                log.debug(f'{tzpath} found, contents:\n {data}')
            for line in data:
                match = zone_re.match(line)
                if match is None:
                    match = timezone_re.match(line)
                if match is not None:
                    line = line[match.end():]
                    etctz = line[:end_re.search(line).start()]
                    found_configs[tzpath] = etctz.replace(' ', '_')
        except (OSError, UnicodeDecodeError):
            continue
    tzpath = os.path.join(_root, 'etc/localtime')
    if os.path.exists(tzpath) and os.path.islink(tzpath):
        log.debug(f'{tzpath} found')
        etctz = os.path.realpath(tzpath)
        start = etctz.find('/') + 1
        while start != 0:
            etctz = etctz[start:]
            try:
                zoneinfo.ZoneInfo(etctz)
                tzinfo = f'{tzpath} is a symlink to'
                found_configs[tzinfo] = etctz.replace(' ', '_')
                break
            except zoneinfo.ZoneInfoNotFoundError:
                pass
            start = etctz.find('/') + 1
    if len(found_configs) > 0:
        log.debug(f'{len(found_configs)} found:\n {found_configs}')
        if len(found_configs) > 1:
            unique_tzs = set()
            zoneinfopath = os.path.join(_root, 'usr', 'share', 'zoneinfo')
            directory_depth = len(zoneinfopath.split(os.path.sep))
            for tzname in found_configs.values():
                path = os.path.realpath(os.path.join(zoneinfopath, *tzname.split('/')))
                real_zone_name = '/'.join(path.split(os.path.sep)[directory_depth:])
                unique_tzs.add(real_zone_name)
            if len(unique_tzs) != 1:
                message = 'Multiple conflicting time zone configurations found:\n'
                for key, value in found_configs.items():
                    message += f'{key}: {value}\n'
                message += 'Fix the configuration, or set the time zone in a TZ environment variable.\n'
                raise zoneinfo.ZoneInfoNotFoundError(message)
        return list(found_configs.values())[0]