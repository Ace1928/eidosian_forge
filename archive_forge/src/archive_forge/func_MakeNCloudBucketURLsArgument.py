from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from .tab_complete import CompleterType
@staticmethod
def MakeNCloudBucketURLsArgument(n):
    """Constructs an argument that takes N Cloud bucket URLs as parameters."""
    return CommandArgument('file', nargs=n, completer=CompleterType.CLOUD_BUCKET)