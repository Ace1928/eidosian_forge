import io
import os
import traceback
from oslo_utils import excutils
from oslo_utils import reflection
class AmbiguousDependency(DependencyFailure):
    """Raised when some type of ambiguous dependency problem occurs."""