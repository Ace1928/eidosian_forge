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
class CommandLine(BaseInterface):
    """Implements functionality to interact with command line programs
    class must be instantiated with a command argument

    Parameters
    ----------
    command : str
        define base immutable `command` you wish to run
    args : str, optional
        optional arguments passed to base `command`

    Examples
    --------
    >>> import pprint
    >>> from nipype.interfaces.base import CommandLine
    >>> cli = CommandLine(command='ls', environ={'DISPLAY': ':1'})
    >>> cli.inputs.args = '-al'
    >>> cli.cmdline
    'ls -al'

    >>> # Use get_traitsfree() to check all inputs set
    >>> pprint.pprint(cli.inputs.get_traitsfree())  # doctest:
    {'args': '-al',
     'environ': {'DISPLAY': ':1'}}

    >>> cli.inputs.get_hashval()[0][0]
    ('args', '-al')
    >>> cli.inputs.get_hashval()[1]
    '11c37f97649cd61627f4afe5136af8c0'

    """
    input_spec = CommandLineInputSpec
    _cmd_prefix = ''
    _cmd = None
    _version = None
    _terminal_output = 'stream'
    _write_cmdline = False

    @classmethod
    def set_default_terminal_output(cls, output_type):
        """Set the default terminal output for CommandLine Interfaces.

        This method is used to set default terminal output for
        CommandLine Interfaces.  However, setting this will not
        update the output type for any existing instances.  For these,
        assign the <instance>.terminal_output.
        """
        if output_type in VALID_TERMINAL_OUTPUT:
            cls._terminal_output = output_type
        else:
            raise AttributeError('Invalid terminal output_type: %s' % output_type)

    def __init__(self, command=None, terminal_output=None, write_cmdline=False, **inputs):
        super(CommandLine, self).__init__(**inputs)
        self._environ = None
        self._cmd = command or getattr(self, '_cmd', None)
        self._ldd = str2bool(config.get('execution', 'get_linked_libs', 'true'))
        if self._cmd is None:
            raise Exception('Missing command')
        if terminal_output is not None:
            self.terminal_output = terminal_output
        self._write_cmdline = write_cmdline

    @property
    def cmd(self):
        """sets base command, immutable"""
        if not self._cmd:
            raise NotImplementedError('CommandLineInterface should wrap an executable, but none has been set.')
        return self._cmd

    @property
    def cmdline(self):
        """`command` plus any arguments (args)
        validates arguments and generates command line"""
        self._check_mandatory_inputs()
        allargs = [self._cmd_prefix + self.cmd] + self._parse_inputs()
        return ' '.join(allargs)

    @property
    def terminal_output(self):
        return self._terminal_output

    @terminal_output.setter
    def terminal_output(self, value):
        if value not in VALID_TERMINAL_OUTPUT:
            raise RuntimeError('Setting invalid value "%s" for terminal_output. Valid values are %s.' % (value, ', '.join(['"%s"' % v for v in VALID_TERMINAL_OUTPUT])))
        self._terminal_output = value

    @property
    def write_cmdline(self):
        return self._write_cmdline

    @write_cmdline.setter
    def write_cmdline(self, value):
        self._write_cmdline = value is True

    def raise_exception(self, runtime):
        raise RuntimeError('Command:\n{cmdline}\nStandard output:\n{stdout}\nStandard error:\n{stderr}\nReturn code: {returncode}'.format(**runtime.dictcopy()))

    def _get_environ(self):
        return getattr(self.inputs, 'environ', {})

    def version_from_command(self, flag='-v', cmd=None):
        iflogger.warning('version_from_command member of CommandLine was Deprecated in nipype-1.0.0 and deleted in 1.1.0')
        if cmd is None:
            cmd = self.cmd.split()[0]
        env = dict(os.environ)
        if which(cmd, env=env):
            out_environ = self._get_environ()
            env.update(out_environ)
            proc = sp.Popen(' '.join((cmd, flag)), shell=True, env=canonicalize_env(env), stdout=sp.PIPE, stderr=sp.PIPE)
            o, e = proc.communicate()
            return o

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        """Execute command via subprocess

        Parameters
        ----------
        runtime : passed by the run function

        Returns
        -------
        runtime :
            updated runtime information
            adds stdout, stderr, merged, cmdline, dependencies, command_path

        """
        out_environ = self._get_environ()
        try:
            runtime.cmdline = self.cmdline
        except Exception as exc:
            raise RuntimeError('Error raised when interpolating the command line') from exc
        runtime.stdout = None
        runtime.stderr = None
        runtime.cmdline = self.cmdline
        runtime.environ.update(out_environ)
        runtime.success_codes = correct_return_codes
        executable_name = shlex.split(self._cmd_prefix + self.cmd)[0]
        cmd_path = which(executable_name, env=runtime.environ)
        if cmd_path is None:
            raise IOError('No command "%s" found on host %s. Please check that the corresponding package is installed.' % (executable_name, runtime.hostname))
        runtime.command_path = cmd_path
        runtime.dependencies = get_dependencies(executable_name, runtime.environ) if self._ldd else '<skipped>'
        runtime = run_command(runtime, output=self.terminal_output, write_cmdline=self.write_cmdline)
        return runtime

    def _format_arg(self, name, trait_spec, value):
        """A helper function for _parse_inputs

        Formats a trait containing argstr metadata
        """
        argstr = trait_spec.argstr
        iflogger.debug('%s_%s', name, value)
        if trait_spec.is_trait_type(traits.Bool) and '%' not in argstr:
            return argstr if value else None
        elif trait_spec.is_trait_type(traits.List) or (trait_spec.is_trait_type(traits.TraitCompound) and isinstance(value, list)):
            sep = trait_spec.sep if trait_spec.sep is not None else ' '
            if argstr.endswith('...'):
                argstr = argstr.replace('...', '')
                return sep.join([argstr % elt for elt in value])
            else:
                return argstr % sep.join((str(elt) for elt in value))
        else:
            return argstr % value

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

    def _gen_filename(self, name):
        raise NotImplementedError

    def _overload_extension(self, value, name=None):
        return value

    def _list_outputs(self):
        metadata = dict(name_source=lambda t: t is not None)
        traits = self.inputs.traits(**metadata)
        if traits:
            outputs = self.output_spec().trait_get()
            for name, trait_spec in list(traits.items()):
                out_name = name
                if trait_spec.output_name is not None:
                    out_name = trait_spec.output_name
                fname = self._filename_from_source(name)
                if isdefined(fname):
                    outputs[out_name] = os.path.abspath(fname)
            return outputs

    def _parse_inputs(self, skip=None):
        """Parse all inputs using the ``argstr`` format string in the Trait.

        Any inputs that are assigned (not the default_value) are formatted
        to be added to the command line.

        Returns
        -------
        all_args : list
            A list of all inputs formatted for the command line.

        """
        all_args = []
        initial_args = {}
        final_args = {}
        metadata = dict(argstr=lambda t: t is not None)
        for name, spec in sorted(self.inputs.traits(**metadata).items()):
            if skip and name in skip:
                continue
            value = getattr(self.inputs, name)
            if spec.name_source:
                value = self._filename_from_source(name)
            elif spec.genfile:
                if not isdefined(value) or value is None:
                    value = self._gen_filename(name)
            if not isdefined(value):
                continue
            try:
                arg = self._format_arg(name, spec, value)
            except Exception as exc:
                raise ValueError(f"Error formatting command line argument '{name}' with value '{value}'") from exc
            if arg is None:
                continue
            pos = spec.position
            if pos is not None:
                if int(pos) >= 0:
                    initial_args[pos] = arg
                else:
                    final_args[pos] = arg
            else:
                all_args.append(arg)
        first_args = [el for _, el in sorted(initial_args.items())]
        last_args = [el for _, el in sorted(final_args.items())]
        return first_args + all_args + last_args