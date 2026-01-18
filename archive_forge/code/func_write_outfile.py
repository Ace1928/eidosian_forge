from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
def write_outfile(self, outdata, outname):
    """Write the output file"""
    if outname:
        print('Writing output to %s' % outname, file=sys.stderr)
        with open(outname, 'wb') as outfile:
            outfile.write(outdata)
    else:
        print('Writing output to stdout', file=sys.stderr)
        rsa._compat.write_to_stdout(outdata)