import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class ACLManager(base.BaseEntityManager):
    """Entity Manager for Secret or Container ACL entities"""
    acl_class_map = {'secret': SecretACL, 'container': ContainerACL}

    def __init__(self, api):
        super(ACLManager, self).__init__(api, ACL._resource_name)

    def create(self, entity_ref=None, users=None, project_access=None, operation_type=DEFAULT_OPERATION_TYPE):
        """Factory method for creating `ACL` entity.

        `ACL` object returned by this method have not yet been
        stored in Barbican.

        Input entity_ref is used to determine whether
        ACL object type needs to be :class:`barbicanclient.acls.SecretACL`
        or  :class:`barbicanclient.acls.ContainerACL`.

        :param str entity_ref: Full HATEOAS reference to a secret or container
        :param users: List of Keystone userid(s) to be used in ACL.
        :type users: List or None
        :param bool project_access: Flag indicating project access behavior
        :param str operation_type: Type indicating which class of Barbican
            operations this ACL is defined for e.g. 'read' operations
        :returns: ACL object instance
        :rtype: :class:`barbicanclient.v1.acls.SecretACL` or
            :class:`barbicanclient.v1.acls.ContainerACL`
        """
        entity_type = ACL.identify_ref_type(entity_ref)
        entity_class = ACLManager.acl_class_map.get(entity_type)
        return entity_class(api=self._api, entity_ref=entity_ref, users=users, project_access=project_access, operation_type=operation_type)

    def get(self, entity_ref):
        """Retrieve existing ACLs for a secret or container found in Barbican

        :param str entity_ref: Full HATEOAS reference to a secret or container.
        :returns: ACL entity object instance
        :rtype: :class:`barbicanclient.v1.acls.SecretACL` or
            :class:`barbicanclient.v1.acls.ContainerACL`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        """
        entity = self._validate_acl_ref(entity_ref)
        LOG.debug('Getting ACL for {0} href: {1}'.format(entity.acl_type, entity.acl_ref))
        entity.load_acls_data()
        return entity

    def _validate_acl_ref(self, entity_ref):
        if entity_ref is None:
            raise ValueError('Expected secret or container URI is not specified.')
        entity_ref = entity_ref.rstrip('/')
        entity_type = ACL.identify_ref_type(entity_ref)
        entity_class = ACLManager.acl_class_map.get(entity_type)
        acl_entity = entity_class(api=self._api, entity_ref=entity_ref)
        acl_entity.validate_input_ref()
        return acl_entity