import os.path
from oslo_log import log
from keystone.catalog.backends import base
from keystone.common import utils
import keystone.conf
from keystone import exception
Retrieve and format the current V3 service catalog.

        This implementation builds the V3 catalog from the V2 catalog.

        :param user_id: The id of the user who has been authenticated for
            creating service catalog.
        :param project_id: The id of the project. 'project_id' will be None in
            the case this being called to create a catalog to go in a domain
            scoped token. In this case, any endpoint that requires a project_id
            as part of their URL will be skipped.

        :returns: A list representing the service catalog or an empty list

        