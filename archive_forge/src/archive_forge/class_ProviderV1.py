from abc import ABC, abstractmethod
from qiskit.providers.exceptions import QiskitBackendNotFoundError
class ProviderV1(Provider, ABC):
    """Base class for a Backend Provider."""
    version = 1

    def get_backend(self, name=None, **kwargs):
        """Return a single backend matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs: dict used for filtering.

        Returns:
            Backend: a backend matching the filtering.

        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
                more than one backend matches the filtering criteria.
        """
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError('More than one backend matches the criteria')
        if not backends:
            raise QiskitBackendNotFoundError('No backend matches the criteria')
        return backends[0]

    @abstractmethod
    def backends(self, name=None, **kwargs):
        """Return a list of backends matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs: dict used for filtering.

        Returns:
            list[Backend]: a list of Backends that match the filtering
                criteria.
        """
        pass

    def __eq__(self, other):
        """Equality comparison.

        By default, it is assumed that two `Providers` from the same class are
        equal. Subclassed providers can override this behavior.
        """
        return type(self).__name__ == type(other).__name__