from keystoneclient import base
class IdentityProviderManager(base.CrudManager):
    """Manager class for manipulating Identity Providers."""
    resource_class = IdentityProvider
    collection_key = 'identity_providers'
    key = 'identity_provider'
    base_url = 'OS-FEDERATION'

    def _build_url_and_put(self, **kwargs):
        url = self.build_url(dict_args_in_out=kwargs)
        body = {self.key: kwargs}
        return self._update(url, body=body, response_key=self.key, method='PUT')

    def create(self, id, **kwargs):
        """Create Identity Provider object.

        Utilize Keystone URI:
        PUT /OS-FEDERATION/identity_providers/$identity_provider

        :param id: unique id of the identity provider.
        :param kwargs: optional attributes: description (str), domain_id (str),
                       enabled (boolean) and remote_ids (list).
        :returns: an IdentityProvider resource object.
        :rtype: :py:class:`keystoneclient.v3.federation.IdentityProvider`

        """
        return self._build_url_and_put(identity_provider_id=id, **kwargs)

    def get(self, identity_provider):
        """Fetch Identity Provider object.

        Utilize Keystone URI:
        GET /OS-FEDERATION/identity_providers/$identity_provider

        :param identity_provider: an object with identity_provider_id
                                  stored inside.
        :returns: an IdentityProvider resource object.
        :rtype: :py:class:`keystoneclient.v3.federation.IdentityProvider`

        """
        return super(IdentityProviderManager, self).get(identity_provider_id=base.getid(identity_provider))

    def list(self, **kwargs):
        """List all Identity Providers.

        Utilize Keystone URI:
        GET /OS-FEDERATION/identity_providers

        :returns: a list of IdentityProvider resource objects.
        :rtype: List

        """
        return super(IdentityProviderManager, self).list(**kwargs)

    def update(self, identity_provider, **kwargs):
        """Update Identity Provider object.

        Utilize Keystone URI:
        PATCH /OS-FEDERATION/identity_providers/$identity_provider

        :param identity_provider: an object with identity_provider_id
                                  stored inside.
        :returns: an IdentityProvider resource object.
        :rtype: :py:class:`keystoneclient.v3.federation.IdentityProvider`

        """
        return super(IdentityProviderManager, self).update(identity_provider_id=base.getid(identity_provider), **kwargs)

    def delete(self, identity_provider):
        """Delete Identity Provider object.

        Utilize Keystone URI:
        DELETE /OS-FEDERATION/identity_providers/$identity_provider

        :param identity_provider: the Identity Provider ID itself or an object
                                  with it stored inside.

        """
        return super(IdentityProviderManager, self).delete(identity_provider_id=base.getid(identity_provider))