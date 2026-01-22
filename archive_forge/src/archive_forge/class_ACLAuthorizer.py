import abc
from ._identity import ACLIdentity
class ACLAuthorizer(Authorizer):
    """ ACLAuthorizer is an Authorizer implementation that will check access
    control list (ACL) membership of users. It uses get_acl to find out
    the ACLs that apply to the requested operations and will authorize an
    operation if an ACL contains the group "everyone" or if the identity is
    an instance of ACLIdentity and its allow method returns True for the ACL.
    """

    def __init__(self, get_acl, allow_public=False):
        """
        :param get_acl get_acl will be called with an auth context and an Op.
        It should return the ACL that applies (an array of string ids).
        If an entity cannot be found or the action is not recognised,
        get_acl should return an empty list but no error.
        :param allow_public: boolean, If True and an ACL contains "everyone",
        then authorization will be granted even if there is no logged in user.
        """
        self._allow_public = allow_public
        self._get_acl = get_acl

    def authorize(self, ctx, identity, ops):
        """Implements Authorizer.authorize by calling identity.allow to
        determine whether the identity is a member of the ACLs associated with
        the given operations.
        """
        if len(ops) == 0:
            return ([], [])
        allowed = [False] * len(ops)
        has_allow = isinstance(identity, ACLIdentity)
        for i, op in enumerate(ops):
            acl = self._get_acl(ctx, op)
            if has_allow:
                allowed[i] = identity.allow(ctx, acl)
            else:
                allowed[i] = self._allow_public and EVERYONE in acl
        return (allowed, [])