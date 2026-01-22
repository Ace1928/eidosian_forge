import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
def _tenant_path(self):
    """URL-encoded path segment of a URL within a particular tenant.

        Returned in the form:
            stacks/<stack_name>/<stack_id><path>
        """
    return 'stacks/%s%s' % (self.stack_path(), urlparse.quote(encodeutils.safe_encode(self.path)))