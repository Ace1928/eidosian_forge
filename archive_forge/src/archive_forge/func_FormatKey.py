from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
def FormatKey(key):
    return "'{key}' has conflicting values '{old}' and '{new}'. Using '{new}'.".format(key=key, old=old_dict[key], new=new_dict[key])