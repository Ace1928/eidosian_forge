import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _parse_cmdline(self, line):
    """
        Parses the command line entered by the user. This is a wrapper around
        the actual pyparsing parser that pre-chews the result trees to
        cleanly extract the tokens we care for (parameters, path, command).
        @param line: The command line to parse.
        @type line: str
        @return: (result_trees, path, command, pparams, kparams),
        pparams being positional parameters and kparams the keyword=value.
        @rtype: (pyparsing.ParseResults, str, str, list, dict)
        """
    self.log.debug('Parsing commandline.')
    path = ''
    command = ''
    pparams = []
    kparams = {}
    parse_results = self._parser.parseString(line)
    if isinstance(parse_results.path, ParseResults):
        path = parse_results.path.value
    if isinstance(parse_results.command, ParseResults):
        command = parse_results.command.value
    if isinstance(parse_results.pparams, ParseResults):
        pparams = [pparam.value for pparam in parse_results.pparams]
    if isinstance(parse_results.kparams, ParseResults):
        kparams = dict([kparam.value for kparam in parse_results.kparams])
    self.log.debug("Parse gave path='%s' command='%s' " % (path, command) + 'pparams=%s ' % str(pparams) + 'kparams=%s' % str(kparams))
    return (parse_results, path, command, pparams, kparams)