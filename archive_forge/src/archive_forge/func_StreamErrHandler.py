from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
def StreamErrHandler(result_holder, capture_output=False):
    """Customized processing for streaming stderr from subprocess."""
    del result_holder, capture_output

    def HandleStdErr(line):
        if line:
            for to_be_ignored in IGNORED_LOGS:
                if to_be_ignored in line:
                    return
            log.status.Print(line)
            if 'server error:' in line and 'bind: address already in use' in line:
                log.status.Print('You can set the --port flag to specify a different local port')
    return HandleStdErr