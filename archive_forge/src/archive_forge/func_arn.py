import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
def arn(self):
    """Return as an ARN.

        Returned in the form:
            arn:openstack:heat::<tenant>:stacks/<stack_name>/<stack_id><path>
        """
    return 'arn:openstack:heat::%s:%s' % (urlparse.quote(self.tenant, ''), self._tenant_path())