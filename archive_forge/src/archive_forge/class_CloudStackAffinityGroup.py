import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackAffinityGroup:
    """
    Class representing a CloudStack AffinityGroup.
    """

    def __init__(self, id, account, description, domain, domainid, name, group_type, virtualmachine_ids):
        """
        A CloudStack Affinity Group.

        @note: This is a non-standard extension API, and only works for
               CloudStack.

        :param      id: CloudStack Affinity Group ID
        :type       id: ``str``

        :param      account: An account for the affinity group. Must be used
                             with domainId.
        :type       account: ``str``

        :param      description: optional description of the affinity group
        :type       description: ``str``

        :param      domain: the domain name of the affinity group
        :type       domain: ``str``

        :param      domainid: domain ID of the account owning the affinity
                              group
        :type       domainid: ``str``

        :param      name: name of the affinity group
        :type       name: ``str``

        :param      group_type: the type of the affinity group
        :type       group_type: :class:`CloudStackAffinityGroupType`

        :param      virtualmachine_ids: virtual machine Ids associated with
                                        this affinity group
        :type       virtualmachine_ids: ``str``

        :rtype:     :class:`CloudStackAffinityGroup`
        """
        self.id = id
        self.account = account
        self.description = description
        self.domain = domain
        self.domainid = domainid
        self.name = name
        self.type = group_type
        self.virtualmachine_ids = virtualmachine_ids

    def __repr__(self):
        return '<CloudStackAffinityGroup: id=%s, name=%s, type=%s>' % (self.id, self.name, self.type)