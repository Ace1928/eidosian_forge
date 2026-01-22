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
def _outputs_from_inputs(self, outputs):
    for name in list(outputs.keys()):
        corresponding_input = getattr(self.inputs, name)
        if isdefined(corresponding_input):
            if isinstance(corresponding_input, bool) and corresponding_input:
                outputs[name] = os.path.abspath(self._outputs_filenames[name])
            elif isinstance(corresponding_input, list):
                outputs[name] = [os.path.abspath(inp) for inp in corresponding_input]
            else:
                outputs[name] = os.path.abspath(corresponding_input)
    return outputs