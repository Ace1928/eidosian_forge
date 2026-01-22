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
def _DictsToFolders(base_path, bucket, flat):
    children = []
    for folder, contents in bucket.items():
        if type(contents) == dict:
            folder_children = _DictsToFolders(os.path.join(base_path, folder), contents, flat)
            if flat:
                children += folder_children
            else:
                folder_children = MSVSNew.MSVSFolder(os.path.join(base_path, folder), name='(' + folder + ')', entries=folder_children)
                children.append(folder_children)
        else:
            children.append(contents)
    return children