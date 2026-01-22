from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def CommandRun(self, argv):
    """Execute the command with given arguments.

    First register and parse additional flags. Then run the command.

    Returns:
      Command return value.

    Args:
      argv: Remaining command line arguments after parsing command and flags
            (that is a copy of sys.argv at the time of the function call with
            all parsed flags removed).
    """
    FLAGS.AppendFlagValues(self._command_flags)
    orig_app_usage = app.usage

    def ReplacementAppUsage(shorthelp=0, writeto_stdout=1, detailed_error=None, exitcode=None):
        AppcommandsUsage(shorthelp, writeto_stdout, detailed_error, exitcode=1, show_cmd=self._command_name, show_global_flags=True)
    app.usage = ReplacementAppUsage
    try:
        try:
            argv = ParseFlagsWithUsage(argv)
            if FLAGS.run_with_pdb:
                ret = pdb.runcall(self.Run, argv)
            else:
                ret = self.Run(argv)
            if ret is None:
                ret = 0
            else:
                assert isinstance(ret, int)
            return ret
        except app.UsageError as error:
            app.usage(shorthelp=1, detailed_error=error, exitcode=error.exitcode)
    finally:
        app.usage = orig_app_usage
        for flag_name in self._command_flags.FlagDict():
            delattr(FLAGS, flag_name)