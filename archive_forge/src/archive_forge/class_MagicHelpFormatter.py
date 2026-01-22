import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
@undoc
class MagicHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """A HelpFormatter with a couple of changes to meet our needs.
    """

    def _fill_text(self, text, width, indent):
        return argparse.RawDescriptionHelpFormatter._fill_text(self, dedent(text), width, indent)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                if not NAME_RE.match(args_string):
                    args_string = '<%s>' % args_string
                for option_string in action.option_strings:
                    parts.append('%s %s' % (option_string, args_string))
            return ', '.join(parts)

    def add_usage(self, usage, actions, groups, prefix='::\n\n  %'):
        super(MagicHelpFormatter, self).add_usage(usage, actions, groups, prefix)