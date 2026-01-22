import getpass
import os
import pdb
import signal
import sys
import traceback
import warnings
from operator import attrgetter
from twisted import copyright, logger, plugin
from twisted.application import reactors, service
from twisted.application.reactors import NoSuchReactor, installReactor
from twisted.internet import defer
from twisted.internet.interfaces import _ISupportsExitSignalCapturing
from twisted.persisted import sob
from twisted.python import failure, log, logfile, runtime, usage, util
from twisted.python.reflect import namedAny, namedModule, qual
class ReactorSelectionMixin:
    """
    Provides options for selecting a reactor to install.

    If a reactor is installed, the short name which was used to locate it is
    saved as the value for the C{"reactor"} key.
    """
    compData = usage.Completions(optActions={'reactor': _reactorAction})
    messageOutput = sys.stdout
    _getReactorTypes = staticmethod(reactors.getReactorTypes)

    def opt_help_reactors(self):
        """
        Display a list of possibly available reactor names.
        """
        rcts = sorted(self._getReactorTypes(), key=attrgetter('shortName'))
        notWorkingReactors = ''
        for r in rcts:
            try:
                namedModule(r.moduleName)
                self.messageOutput.write(f'    {r.shortName:<4}\t{r.description}\n')
            except ImportError as e:
                notWorkingReactors += '    !{:<4}\t{} ({})\n'.format(r.shortName, r.description, e.args[0])
        if notWorkingReactors:
            self.messageOutput.write('\n')
            self.messageOutput.write('    reactors not available on this platform:\n\n')
            self.messageOutput.write(notWorkingReactors)
        raise SystemExit(0)

    def opt_reactor(self, shortName):
        """
        Which reactor to use (see --help-reactors for a list of possibilities)
        """
        try:
            installReactor(shortName)
        except NoSuchReactor:
            msg = "The specified reactor does not exist: '%s'.\nSee the list of available reactors with --help-reactors" % (shortName,)
            raise usage.UsageError(msg)
        except Exception as e:
            msg = 'The specified reactor cannot be used, failed with error: %s.\nSee the list of available reactors with --help-reactors' % (e,)
            raise usage.UsageError(msg)
        else:
            self['reactor'] = shortName
    opt_r = opt_reactor