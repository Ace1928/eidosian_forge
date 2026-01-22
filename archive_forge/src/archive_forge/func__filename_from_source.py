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
def _filename_from_source(self, name, chain=None):
    if chain is None:
        chain = []
    trait_spec = self.inputs.trait(name)
    retval = getattr(self.inputs, name)
    source_ext = None
    if not isdefined(retval) or '%s' in retval:
        if not trait_spec.name_source:
            return retval
        if any((isdefined(getattr(self.inputs, field)) for field in trait_spec.xor or ())):
            return retval
        if not all((isdefined(getattr(self.inputs, field)) for field in trait_spec.requires or ())):
            return retval
        if isdefined(retval) and '%s' in retval:
            name_template = retval
        else:
            name_template = trait_spec.name_template
        if not name_template:
            name_template = '%s_generated'
        ns = trait_spec.name_source
        while isinstance(ns, (list, tuple)):
            if len(ns) > 1:
                iflogger.warning('Only one name_source per trait is allowed')
            ns = ns[0]
        if not isinstance(ns, (str, bytes)):
            raise ValueError("name_source of '{}' trait should be an input trait name, but a type {} object was found".format(name, type(ns)))
        if isdefined(getattr(self.inputs, ns)):
            name_source = ns
            source = getattr(self.inputs, name_source)
            while isinstance(source, list):
                source = source[0]
            try:
                _, base, source_ext = split_filename(source)
            except (AttributeError, TypeError):
                base = source
        else:
            if name in chain:
                raise NipypeInterfaceError('Mutually pointing name_sources')
            chain.append(name)
            base = self._filename_from_source(ns, chain)
            if isdefined(base):
                _, _, source_ext = split_filename(base)
            else:
                return retval
        chain = None
        retval = name_template % base
        _, _, ext = split_filename(retval)
        if trait_spec.keep_extension and (ext or source_ext):
            if (ext is None or not ext) and source_ext:
                retval = retval + source_ext
        else:
            retval = self._overload_extension(retval, name)
    return retval