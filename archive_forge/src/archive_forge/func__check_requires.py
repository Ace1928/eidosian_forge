import os
import subprocess as sp
import shlex
import simplejson as json
from traits.trait_errors import TraitError
from ... import config, logging, LooseVersion
from ...utils.provenance import write_provenance
from ...utils.misc import str2bool
from ...utils.filemanip import (
from ...utils.subprocess import run_command
from ...external.due import due
from .traits_extension import traits, isdefined, Undefined
from .specs import (
from .support import (
def _check_requires(self, spec, name, value):
    """check if required inputs are satisfied"""
    if spec.requires:
        values = [not isdefined(getattr(self.inputs, field)) for field in spec.requires]
        if any(values) and isdefined(value):
            if len(values) > 1:
                fmt = "%s requires values for inputs %s because '%s' is set. For a list of required inputs, see %s.help()"
            else:
                fmt = "%s requires a value for input %s because '%s' is set. For a list of required inputs, see %s.help()"
            msg = fmt % (self.__class__.__name__, ', '.join(("'%s'" % req for req in spec.requires)), name, self.__class__.__name__)
            raise ValueError(msg)