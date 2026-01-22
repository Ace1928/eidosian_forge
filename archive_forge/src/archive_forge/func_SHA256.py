from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
def SHA256(content, prefix='sha256:'):
    """Return 'sha256:' + hex(sha256(content))."""
    return prefix + hashlib.sha256(content).hexdigest()