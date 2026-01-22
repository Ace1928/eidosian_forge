import sys
from enchant.checker import SpellChecker
def _run_as_script():
    """Run the command-line spellchecker as a script.
    This function allows the spellchecker to be invoked from the command-line
    to check spelling in a file.
    """
    from optparse import OptionParser
    op = OptionParser()
    op.add_option('-o', '--output', dest='outfile', metavar='FILE', help='write changes into FILE')
    op.add_option('-l', '--lang', dest='lang', metavar='TAG', default='en_US', help='use language idenfified by TAG')
    op.add_option('-e', '--encoding', dest='enc', metavar='ENC', help='file is unicode with encoding ENC')
    opts, args = op.parse_args()
    if len(args) < 1:
        raise ValueError('Must name a file to check')
    if len(args) > 1:
        raise ValueError('Can only check a single file')
    chkr = SpellChecker(opts.lang)
    cmdln = CmdLineChecker()
    cmdln.set_checker(chkr)
    cmdln.run_on_file(args[0], opts.outfile, opts.enc)