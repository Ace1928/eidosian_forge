from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting_windows  # pylint: disable=unused-import
import termcolor
def Bold(text):
    return termcolor.colored(text, attrs=['bold'])