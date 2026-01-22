import os
from typing import List
from .. import forge as _mod_forge
from .. import registry, tests, urlutils
from ..forge import (Forge, MergeProposal, UnsupportedForge, determine_title,
@classmethod
def _add_location(cls, url):
    cls._locations.append(url)