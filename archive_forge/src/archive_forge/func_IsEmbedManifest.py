import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def IsEmbedManifest(self, config):
    """Returns whether manifest should be linked into binary."""
    config = self._TargetConfig(config)
    embed = self._Setting(('VCManifestTool', 'EmbedManifest'), config, default='true')
    return embed == 'true'