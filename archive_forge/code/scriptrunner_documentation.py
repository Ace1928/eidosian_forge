import os
import sys
from os.path import dirname, isdir, isfile
from os.path import join as pjoin
from os.path import pathsep, realpath
from subprocess import PIPE, Popen
Run command sequence `cmd` returning exit code, stdout, stderr

        Parameters
        ----------
        cmd : str or sequence
            string with command name or sequence of strings defining command
        check_code : {True, False}, optional
            If True, raise error for non-zero return code

        Returns
        -------
        returncode : int
            return code from execution of `cmd`
        stdout : bytes
            stdout from `cmd`
        stderr : bytes
            stderr from `cmd`
        