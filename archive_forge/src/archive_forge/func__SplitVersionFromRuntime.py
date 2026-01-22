import re
from googlecloudsdk.command_lib.run import exceptions
def _SplitVersionFromRuntime(runtime_language):
    return re.sub('[0-9]+$', '', runtime_language)