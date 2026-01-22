import os
import sys
import time
from rdkit import RDConfig
def ReportResults(script, failedTests, nTests, runTime, verbose, dest):
    if not nTests:
        dest.write('!-!-!-!-!-!-!-!-!-!-!\n')
        dest.write('\tScript: %s.  No tests run!\n' % script)
    elif not len(failedTests):
        dest.write('-----------------\n')
        dest.write('\tScript: %s.  Passed %d tests in %.2f seconds\n' % (script, nTests, runTime))
    else:
        dest.write('!-!-!-!-!-!-!-!-!-!-!\n')
        dest.write('\tScript: %s.  Failed %d (of %d) tests in %.2f seconds\n' % (script, len(failedTests), nTests, runTime))
        if verbose:
            for exeName, args, extras in failedTests:
                dirName = extras.get('dir', '.')
                dirName = os.path.abspath(dirName)
                dest.write('\t\t(%s): %s %s\n' % (dirName, exeName, args))