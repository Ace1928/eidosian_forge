import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def VerifyMissingSources(sources, build_dir, generator_flags, gyp_to_ninja):
    """Emulate behavior of msvs_error_on_missing_sources present in the msvs
    generator: Check that all regular source files, i.e. not created at run time,
    exist on disk. Missing files cause needless recompilation when building via
    VS, and we want this check to match for people/bots that build using ninja,
    so they're not surprised when the VS build fails."""
    if int(generator_flags.get('msvs_error_on_missing_sources', 0)):
        no_specials = filter(lambda x: '$' not in x, sources)
        relative = [os.path.join(build_dir, gyp_to_ninja(s)) for s in no_specials]
        missing = [x for x in relative if not os.path.exists(x)]
        if missing:
            cleaned_up = [os.path.normpath(x) for x in missing]
            raise Exception('Missing input files:\n%s' % '\n'.join(cleaned_up))