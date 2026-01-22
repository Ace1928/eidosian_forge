import os
from .. import config
from .base import (
class MatlabCommand(CommandLine):
    """Interface that runs matlab code

    >>> import nipype.interfaces.matlab as matlab
    >>> mlab = matlab.MatlabCommand(mfile=False)  # don't write script file
    >>> mlab.inputs.script = "which('who')"
    >>> out = mlab.run()  # doctest: +SKIP
    """
    _cmd = 'matlab'
    _default_matlab_cmd = None
    _default_mfile = None
    _default_paths = None
    input_spec = MatlabInputSpec

    def __init__(self, matlab_cmd=None, **inputs):
        """initializes interface to matlab
        (default 'matlab -nodesktop -nosplash')
        """
        super(MatlabCommand, self).__init__(**inputs)
        if matlab_cmd and isdefined(matlab_cmd):
            self._cmd = matlab_cmd
        elif self._default_matlab_cmd:
            self._cmd = self._default_matlab_cmd
        if self._default_mfile and (not isdefined(self.inputs.mfile)):
            self.inputs.mfile = self._default_mfile
        if self._default_paths and (not isdefined(self.inputs.paths)):
            self.inputs.paths = self._default_paths
        if not isdefined(self.inputs.single_comp_thread) and (not isdefined(self.inputs.uses_mcr)):
            if config.getboolean('execution', 'single_thread_matlab'):
                self.inputs.single_comp_thread = True
        self.terminal_output = 'allatonce'

    @classmethod
    def set_default_matlab_cmd(cls, matlab_cmd):
        """Set the default MATLAB command line for MATLAB classes.

        This method is used to set values for all MATLAB
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.matlab_cmd.
        """
        cls._default_matlab_cmd = matlab_cmd

    @classmethod
    def set_default_mfile(cls, mfile):
        """Set the default MATLAB script file format for MATLAB classes.

        This method is used to set values for all MATLAB
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.mfile.
        """
        cls._default_mfile = mfile

    @classmethod
    def set_default_paths(cls, paths):
        """Set the default MATLAB paths for MATLAB classes.

        This method is used to set values for all MATLAB
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.paths.
        """
        cls._default_paths = paths

    def _run_interface(self, runtime):
        self.terminal_output = 'allatonce'
        runtime = super(MatlabCommand, self)._run_interface(runtime)
        try:
            os.system('stty sane')
        except:
            pass
        if 'MATLAB code threw an exception' in runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _format_arg(self, name, trait_spec, value):
        if name in ['script']:
            argstr = trait_spec.argstr
            if self.inputs.uses_mcr:
                argstr = '%s'
            return self._gen_matlab_command(argstr, value)
        return super(MatlabCommand, self)._format_arg(name, trait_spec, value)

    def _gen_matlab_command(self, argstr, script_lines):
        """Generates commands and, if mfile specified, writes it to disk."""
        cwd = os.getcwd()
        mfile = self.inputs.mfile or self.inputs.uses_mcr
        paths = []
        if isdefined(self.inputs.paths):
            paths = self.inputs.paths
        prescript = self.inputs.prescript
        postscript = self.inputs.postscript
        if mfile:
            prescript.insert(0, "fprintf(1,'Executing %s at %s:\\n',mfilename(),datestr(now));")
        else:
            prescript.insert(0, "fprintf(1,'Executing code at %s:\\n',datestr(now));")
        for path in paths:
            prescript.append("if ~(ismcc || isdeployed), addpath('%s'); end;\n" % path)
        if not mfile:
            script_lines = ','.join([line for line in script_lines.split('\n') if not line.strip().startswith('%')])
        script_lines = '\n'.join(prescript) + script_lines + '\n'.join(postscript)
        if mfile:
            with open(os.path.join(cwd, self.inputs.script_file), 'wt') as mfile:
                mfile.write(script_lines)
            if self.inputs.uses_mcr:
                script = '%s' % os.path.join(cwd, self.inputs.script_file)
            else:
                script = "addpath('%s');%s" % (cwd, self.inputs.script_file.split('.')[0])
        else:
            script = ''.join(script_lines.split('\n'))
        return argstr % script