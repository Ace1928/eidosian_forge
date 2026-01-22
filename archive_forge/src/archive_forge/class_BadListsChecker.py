from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
class BadListsChecker(Checker):
    """Checks command flags that take lists."""
    name = 'BadLists'
    description = 'Verifies all flags implement lists properly.'

    def __init__(self):
        super(BadListsChecker, self).__init__()
        self._issues = []

    def _ForEvery(self, cmd_or_group):
        for flag in cmd_or_group.GetSpecificFlags():
            if flag.nargs not in [None, 0, 1]:
                self._issues.append(LintError(name=BadListsChecker.name, command=cmd_or_group, error_message='flag [{flg}] has nargs={nargs}'.format(flg=flag.option_strings[0], nargs="'{}'".format(six.text_type(flag.nargs)))))
            if isinstance(flag.type, arg_parsers.ArgDict):
                if not (flag.metavar or flag.type.spec):
                    self._issues.append(LintError(name=BadListsChecker.name, command=cmd_or_group, error_message='dict flag [{flg}] has no metavar and type.spec (at least one needed)'.format(flg=flag.option_strings[0])))
            elif isinstance(flag.type, arg_parsers.ArgList):
                if not flag.metavar:
                    self._issues.append(LintError(name=BadListsChecker.name, command=cmd_or_group, error_message='list flag [{flg}] has no metavar'.format(flg=flag.option_strings[0])))

    def ForEveryGroup(self, group):
        self._ForEvery(group)

    def ForEveryCommand(self, command):
        self._ForEvery(command)

    def End(self):
        return self._issues