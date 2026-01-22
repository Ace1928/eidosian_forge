from monascaclient.common import monasca_manager
class NotificationTypesManager(monasca_manager.MonascaManager):
    base_url = '/notification-methods/types'

    def list(self, **kwargs):
        """Get a list of notifications."""
        return self._list('', **kwargs)