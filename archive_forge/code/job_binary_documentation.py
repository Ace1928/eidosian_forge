from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import support
A resource for creating sahara job binary.

    A job binary stores an URL to a single script or Jar file and any
    credentials needed to retrieve the file.
    