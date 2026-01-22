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
def _GetMSBuildPropertyGroup(spec, label, properties):
    """Returns a PropertyGroup definition for the specified properties.

  Arguments:
    spec: The target project dict.
    label: An optional label for the PropertyGroup.
    properties: The dictionary to be converted.  The key is the name of the
        property.  The value is itself a dictionary; its key is the value and
        the value a list of condition for which this value is true.
  """
    group = ['PropertyGroup']
    if label:
        group.append({'Label': label})
    num_configurations = len(spec['configurations'])

    def GetEdges(node):
        edges = set()
        for value in sorted(properties[node].keys()):
            edges.update({v for v in MSVS_VARIABLE_REFERENCE.findall(value) if v in properties and v != node})
        return edges
    properties_ordered = gyp.common.TopologicallySorted(properties.keys(), GetEdges)
    for name in reversed(properties_ordered):
        values = properties[name]
        for value, conditions in sorted(values.items()):
            if len(conditions) == num_configurations:
                group.append([name, value])
            else:
                for condition in conditions:
                    group.append([name, {'Condition': condition}, value])
    return [group]