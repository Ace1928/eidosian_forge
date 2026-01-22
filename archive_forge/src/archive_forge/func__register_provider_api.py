def _register_provider_api(self, name, obj):
    """Register an instance of a class as a provider api."""
    if name == 'driver':
        raise ValueError('A provider may not be named "driver".')
    if self.locked:
        raise RuntimeError('Programming Error: The provider api registry has been locked (post configuration). Ensure all provider api managers are instantiated before locking.')
    if name in self.__registry:
        raise DuplicateProviderError('`%(name)s` has already been registered as an api provider by `%(prov)r`' % {'name': name, 'prov': self.__registry[name]})
    self.__registry[name] = obj