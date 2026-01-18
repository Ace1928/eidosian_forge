from openstack.key_manager.v1 import container as _container
from openstack.key_manager.v1 import order as _order
from openstack.key_manager.v1 import secret as _secret
from openstack import proxy
def update_secret(self, secret, **attrs):
    """Update a secret

        :param secret: Either the id of a secret or a
            :class:`~openstack.key_manager.v1.secret.Secret` instance.
        :param attrs: The attributes to update on the secret represented
            by ``secret``.

        :returns: The updated secret
        :rtype: :class:`~openstack.key_manager.v1.secret.Secret`
        """
    return self._update(_secret.Secret, secret, **attrs)