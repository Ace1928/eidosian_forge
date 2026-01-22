from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
def _EmbedCode(variables):
    import code
    code.InteractiveConsole(variables).interact()