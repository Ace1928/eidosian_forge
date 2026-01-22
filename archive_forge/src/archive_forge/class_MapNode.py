from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
class MapNode(Node):
    """Wraps interface objects that need to be iterated on a list of inputs.

    Examples
    --------

    >>> from nipype import MapNode
    >>> from nipype.interfaces import fsl
    >>> realign = MapNode(fsl.MCFLIRT(), 'in_file', 'realign')
    >>> realign.inputs.in_file = ['functional.nii',
    ...                           'functional2.nii',
    ...                           'functional3.nii']
    >>> realign.run() # doctest: +SKIP

    """

    def __init__(self, interface, iterfield, name, serial=False, nested=False, **kwargs):
        """

        Parameters
        ----------
        interface : interface object
            node specific interface (fsl.Bet(), spm.Coregister())
        iterfield : string or list of strings
            name(s) of input fields that will receive a list of whatever kind
            of input they take. the node will be run separately for each
            value in these lists. for more than one input, the values are
            paired (i.e. it does not compute a combinatorial product).
        name : alphanumeric string
            node specific name
        serial : boolean
            flag to enforce executing the jobs of the mapnode in a serial
            manner rather than parallel
        nested : boolean
            support for nested lists. If set, the input list will be flattened
            before running and the nested list structure of the outputs will
            be resored.

        See Node docstring for additional keyword arguments.
        """
        super(MapNode, self).__init__(interface, name, **kwargs)
        if isinstance(iterfield, (str, bytes)):
            iterfield = [iterfield]
        self.iterfield = iterfield
        self.nested = nested
        self._inputs = self._create_dynamic_traits(self._interface.inputs, fields=self.iterfield)
        self._inputs.on_trait_change(self._set_mapnode_input)
        self._got_inputs = False
        self._serial = serial

    def _create_dynamic_traits(self, basetraits, fields=None, nitems=None):
        """Convert specific fields of a trait to accept multiple inputs"""
        output = DynamicTraitedSpec()
        if fields is None:
            fields = basetraits.copyable_trait_names()
        for name, spec in list(basetraits.items()):
            if name in fields and (nitems is None or nitems > 1):
                logger.debug('adding multipath trait: %s', name)
                if self.nested:
                    output.add_trait(name, InputMultiPath(traits.Any()))
                else:
                    output.add_trait(name, InputMultiPath(spec.trait_type))
            else:
                output.add_trait(name, traits.Trait(spec))
            setattr(output, name, Undefined)
            value = getattr(basetraits, name)
            if isdefined(value):
                setattr(output, name, value)
            value = getattr(output, name)
        return output

    def set_input(self, parameter, val):
        """
        Set interface input value or nodewrapper attribute
        Priority goes to interface.
        """
        logger.debug('setting nodelevel(%s) input %s = %s', str(self), parameter, str(val))
        self._set_mapnode_input(parameter, deepcopy(val))

    def _set_mapnode_input(self, name, newvalue):
        logger.debug('setting mapnode(%s) input: %s -> %s', str(self), name, str(newvalue))
        if name in self.iterfield:
            setattr(self._inputs, name, newvalue)
        else:
            setattr(self._interface.inputs, name, newvalue)

    def _get_hashval(self):
        """Compute hash including iterfield lists."""
        self._get_inputs()
        if self._hashvalue is not None and self._hashed_inputs is not None:
            return (self._hashed_inputs, self._hashvalue)
        self._check_iterfield()
        hashinputs = deepcopy(self._interface.inputs)
        for name in self.iterfield:
            hashinputs.remove_trait(name)
            hashinputs.add_trait(name, InputMultiPath(self._interface.inputs.traits()[name].trait_type))
            logger.debug('setting hashinput %s-> %s', name, getattr(self._inputs, name))
            if self.nested:
                setattr(hashinputs, name, flatten(getattr(self._inputs, name)))
            else:
                setattr(hashinputs, name, getattr(self._inputs, name))
        hashed_inputs, hashvalue = hashinputs.get_hashval(hash_method=self.config['execution']['hash_method'])
        rm_extra = self.config['execution']['remove_unnecessary_outputs']
        if str2bool(rm_extra) and self.needed_outputs:
            hashobject = md5()
            hashobject.update(hashvalue.encode())
            sorted_outputs = sorted(self.needed_outputs)
            hashobject.update(str(sorted_outputs).encode())
            hashvalue = hashobject.hexdigest()
            hashed_inputs.append(('needed_outputs', sorted_outputs))
        self._hashed_inputs, self._hashvalue = (hashed_inputs, hashvalue)
        return (self._hashed_inputs, self._hashvalue)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        if self._interface._outputs():
            return Bunch(self._interface._outputs().trait_get())

    def _make_nodes(self, cwd=None):
        if cwd is None:
            cwd = self.output_dir()
        if self.nested:
            nitems = len(flatten(ensure_list(getattr(self.inputs, self.iterfield[0]))))
        else:
            nitems = len(ensure_list(getattr(self.inputs, self.iterfield[0])))
        for i in range(nitems):
            nodename = '_%s%d' % (self.name, i)
            node = Node(deepcopy(self._interface), n_procs=self._n_procs, mem_gb=self._mem_gb, overwrite=self.overwrite, needed_outputs=self.needed_outputs, run_without_submitting=self.run_without_submitting, base_dir=op.join(cwd, 'mapflow'), name=nodename)
            node.plugin_args = self.plugin_args
            node.interface.inputs.trait_set(**deepcopy(self._interface.inputs.trait_get()))
            node.interface.resource_monitor = self._interface.resource_monitor
            for field in self.iterfield:
                if self.nested:
                    fieldvals = flatten(ensure_list(getattr(self.inputs, field)))
                else:
                    fieldvals = ensure_list(getattr(self.inputs, field))
                logger.debug('setting input %d %s %s', i, field, fieldvals[i])
                setattr(node.inputs, field, fieldvals[i])
            node.config = self.config
            yield (i, node)

    def _collate_results(self, nodes):
        finalresult = InterfaceResult(interface=[], runtime=[], provenance=[], inputs=[], outputs=self.outputs)
        returncode = []
        for i, nresult, err in nodes:
            finalresult.runtime.insert(i, None)
            returncode.insert(i, err)
            if nresult:
                if hasattr(nresult, 'runtime'):
                    finalresult.interface.insert(i, nresult.interface)
                    finalresult.inputs.insert(i, nresult.inputs)
                    finalresult.runtime[i] = nresult.runtime
                if hasattr(nresult, 'provenance'):
                    finalresult.provenance.insert(i, nresult.provenance)
            if self.outputs:
                for key, _ in list(self.outputs.items()):
                    rm_extra = self.config['execution']['remove_unnecessary_outputs']
                    if str2bool(rm_extra) and self.needed_outputs:
                        if key not in self.needed_outputs:
                            continue
                    values = getattr(finalresult.outputs, key)
                    if not isdefined(values):
                        values = []
                    if nresult and nresult.outputs:
                        values.insert(i, nresult.outputs.trait_get()[key])
                    else:
                        values.insert(i, None)
                    defined_vals = [isdefined(val) for val in values]
                    if any(defined_vals) and finalresult.outputs:
                        setattr(finalresult.outputs, key, values)
        if self.nested:
            for key, _ in list(self.outputs.items()):
                values = getattr(finalresult.outputs, key)
                if isdefined(values):
                    values = unflatten(values, ensure_list(getattr(self.inputs, self.iterfield[0])))
                setattr(finalresult.outputs, key, values)
        if returncode and any([code is not None for code in returncode]):
            msg = []
            for i, code in enumerate(returncode):
                if code is not None:
                    msg += ['Subnode %d failed' % i]
                    msg += ['Error: %s' % str(code)]
            raise NodeExecutionError('Subnodes of node: %s failed:\n%s' % (self.name, '\n'.join(msg)))
        return finalresult

    def get_subnodes(self):
        """Generate subnodes of a mapnode and write pre-execution report"""
        self._get_inputs()
        self._check_iterfield()
        write_node_report(self, result=None, is_mapnode=True)
        return [node for _, node in self._make_nodes()]

    def num_subnodes(self):
        """Get the number of subnodes to iterate in this MapNode"""
        self._get_inputs()
        self._check_iterfield()
        if self._serial:
            return 1
        if self.nested:
            return len(ensure_list(flatten(getattr(self.inputs, self.iterfield[0]))))
        return len(ensure_list(getattr(self.inputs, self.iterfield[0])))

    def _get_inputs(self):
        old_inputs = self._inputs.trait_get()
        self._inputs = self._create_dynamic_traits(self._interface.inputs, fields=self.iterfield)
        self._inputs.trait_set(**old_inputs)
        super(MapNode, self)._get_inputs()

    def _check_iterfield(self):
        """Checks iterfield

        * iterfield must be in inputs
        * number of elements must match across iterfield
        """
        for iterfield in self.iterfield:
            if not isdefined(getattr(self.inputs, iterfield)):
                raise ValueError('Input %s was not set but it is listed in iterfields.' % iterfield)
        if len(self.iterfield) > 1:
            first_len = len(ensure_list(getattr(self.inputs, self.iterfield[0])))
            for iterfield in self.iterfield[1:]:
                if first_len != len(ensure_list(getattr(self.inputs, iterfield))):
                    raise ValueError('All iterfields of a MapNode have to have the same length. %s' % str(self.inputs))

    def _run_interface(self, execute=True, updatehash=False):
        """Run the mapnode interface

        This is primarily intended for serial execution of mapnode. A parallel
        execution requires creation of new nodes that can be spawned
        """
        self._check_iterfield()
        cwd = self.output_dir()
        if not execute:
            return self._load_results()
        if self.nested:
            nitems = len(ensure_list(flatten(getattr(self.inputs, self.iterfield[0]))))
        else:
            nitems = len(ensure_list(getattr(self.inputs, self.iterfield[0])))
        nnametpl = '_%s{}' % self.name
        nodenames = [nnametpl.format(i) for i in range(nitems)]
        outdir = self.output_dir()
        result = InterfaceResult(interface=self._interface.__class__, runtime=Bunch(cwd=outdir, returncode=1, environ=dict(os.environ), hostname=socket.gethostname()), inputs=self._interface.inputs.get_traitsfree())
        try:
            result = self._collate_results(_node_runner(self._make_nodes(cwd), updatehash=updatehash, stop_first=str2bool(self.config['execution']['stop_on_first_crash'])))
        except Exception as msg:
            result.runtime.stderr = '%s\n\n%s'.format(getattr(result.runtime, 'stderr', ''), msg)
            _save_resultfile(result, outdir, self.name, rebase=str2bool(self.config['execution']['use_relative_paths']))
            raise
        _save_resultfile(result, cwd, self.name, rebase=False)
        dirs2remove = []
        for path in glob(op.join(cwd, 'mapflow', '*')):
            if op.isdir(path):
                if path.split(op.sep)[-1] not in nodenames:
                    dirs2remove.append(path)
        for path in dirs2remove:
            logger.debug('[MapNode] Removing folder "%s".', path)
            shutil.rmtree(path)
        return result