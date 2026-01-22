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
def _GetCopies(spec):
    copies = []
    for cpy in spec.get('copies', []):
        for src in cpy.get('files', []):
            dst = os.path.join(cpy['destination'], os.path.basename(src))
            if src.endswith('/'):
                src_bare = src[:-1]
                base_dir = posixpath.split(src_bare)[0]
                outer_dir = posixpath.split(src_bare)[1]
                fixed_dst = _FixPath(dst)
                full_dst = f'"{fixed_dst}\\{outer_dir}\\"'
                cmd = 'mkdir {} 2>nul & cd "{}" && xcopy /e /f /y "{}" {}'.format(full_dst, _FixPath(base_dir), outer_dir, full_dst)
                copies.append(([src], ['dummy_copies', dst], cmd, f'Copying {src} to {fixed_dst}'))
            else:
                fix_dst = _FixPath(cpy['destination'])
                cmd = 'mkdir "{}" 2>nul & set ERRORLEVEL=0 & copy /Y "{}" "{}"'.format(fix_dst, _FixPath(src), _FixPath(dst))
                copies.append(([src], [dst], cmd, f'Copying {src} to {fix_dst}'))
    return copies