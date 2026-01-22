from monascaclient.common import monasca_manager
class NotificationsManager(monasca_manager.MonascaManager):
    base_url = '/notification-methods'

    def create(self, **kwargs):
        """Create a notification."""
        body = self.client.create(url=self.base_url, json=kwargs)
        return body

    def get(self, **kwargs):
        """Get the details for a specific notification."""
        url = '%s/%s' % (self.base_url, kwargs['notification_id'])
        resp = self.client.list(path=url)
        return resp

    def list(self, **kwargs):
        """Get a list of notifications."""
        return self._list('', **kwargs)

    def delete(self, **kwargs):
        """Delete a notification."""
        url = self.base_url + '/%s' % kwargs['notification_id']
        resp = self.client.delete(url=url)
        return resp

    def update(self, **kwargs):
        """Update a notification."""
        url_str = self.base_url + '/%s' % kwargs['notification_id']
        del kwargs['notification_id']
        resp = self.client.create(url=url_str, method='PUT', json=kwargs)
        return resp

    def patch(self, **kwargs):
        """Patch a notification."""
        url_str = self.base_url + '/%s' % kwargs['notification_id']
        del kwargs['notification_id']
        resp = self.client.create(url=url_str, method='PATCH', json=kwargs)
        return resp