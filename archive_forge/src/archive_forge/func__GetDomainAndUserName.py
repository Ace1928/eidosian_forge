import ntpath
import os
import posixpath
import re
import subprocess
import sys
from collections import OrderedDict
import gyp.common
import gyp.easy_xml as easy_xml
import gyp.generator.ninja as ninja_generator
import gyp.MSVSNew as MSVSNew
import gyp.MSVSProject as MSVSProject
import gyp.MSVSSettings as MSVSSettings
import gyp.MSVSToolFile as MSVSToolFile
import gyp.MSVSUserFile as MSVSUserFile
import gyp.MSVSUtil as MSVSUtil
import gyp.MSVSVersion as MSVSVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def _GetDomainAndUserName():
    if sys.platform not in ('win32', 'cygwin'):
        return ('DOMAIN', 'USERNAME')
    global cached_username
    global cached_domain
    if not cached_domain or not cached_username:
        domain = os.environ.get('USERDOMAIN')
        username = os.environ.get('USERNAME')
        if not domain or not username:
            call = subprocess.Popen(['net', 'config', 'Workstation'], stdout=subprocess.PIPE)
            config = call.communicate()[0].decode('utf-8')
            username_re = re.compile('^User name\\s+(\\S+)', re.MULTILINE)
            username_match = username_re.search(config)
            if username_match:
                username = username_match.group(1)
            domain_re = re.compile('^Logon domain\\s+(\\S+)', re.MULTILINE)
            domain_match = domain_re.search(config)
            if domain_match:
                domain = domain_match.group(1)
        cached_domain = domain
        cached_username = username
    return (cached_domain, cached_username)