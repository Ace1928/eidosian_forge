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
def _PrepareListOfSources(spec, generator_flags, gyp_file):
    """Prepare list of sources and excluded sources.

  Besides the sources specified directly in the spec, adds the gyp file so
  that a change to it will cause a re-compile. Also adds appropriate sources
  for actions and copies. Assumes later stage will un-exclude files which
  have custom build steps attached.

  Arguments:
    spec: The target dictionary containing the properties of the target.
    gyp_file: The name of the gyp file.
  Returns:
    A pair of (list of sources, list of excluded sources).
    The sources will be relative to the gyp file.
  """
    sources = OrderedSet()
    _AddNormalizedSources(sources, spec.get('sources', []))
    excluded_sources = OrderedSet()
    if not generator_flags.get('standalone'):
        sources.add(gyp_file)
    for a in spec.get('actions', []):
        inputs = a['inputs']
        inputs = [_NormalizedSource(i) for i in inputs]
        inputs = OrderedSet(inputs)
        sources.update(inputs)
        if not spec.get('msvs_external_builder'):
            excluded_sources.update(inputs)
        if int(a.get('process_outputs_as_sources', False)):
            _AddNormalizedSources(sources, a.get('outputs', []))
    for cpy in spec.get('copies', []):
        _AddNormalizedSources(sources, cpy.get('files', []))
    return (sources, excluded_sources)