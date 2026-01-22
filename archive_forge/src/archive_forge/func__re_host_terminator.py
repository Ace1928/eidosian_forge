from uc_micro.categories import Cc, Cf, P, Z
from uc_micro.properties import Any
def _re_host_terminator(opts):
    src_host_terminator = '(?=$|' + TEXT_SEPARATORS + '|' + SRC_ZPCC + ')' + '(?!' + ('-(?!--)|' if opts.get('---') else '-|') + '_|:\\d|\\.-|\\.(?!$|' + SRC_ZPCC + '))'
    return src_host_terminator