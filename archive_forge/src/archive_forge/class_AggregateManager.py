from novaclient import api_versions
from novaclient import base
class AggregateManager(base.ManagerWithFind):
    resource_class = Aggregate

    def list(self):
        """Get a list of os-aggregates."""
        return self._list('/os-aggregates', 'aggregates')

    def create(self, name, availability_zone):
        """Create a new aggregate."""
        body = {'aggregate': {'name': name, 'availability_zone': availability_zone}}
        return self._create('/os-aggregates', body, 'aggregate')

    def get(self, aggregate):
        """Get details of the specified aggregate."""
        return self._get('/os-aggregates/%s' % base.getid(aggregate), 'aggregate')

    def get_details(self, aggregate):
        """Get details of the specified aggregate."""
        return self.get(aggregate)

    def update(self, aggregate, values):
        """Update the name and/or availability zone."""
        body = {'aggregate': values}
        return self._update('/os-aggregates/%s' % base.getid(aggregate), body, 'aggregate')

    def add_host(self, aggregate, host):
        """Add a host into the Host Aggregate."""
        body = {'add_host': {'host': host}}
        return self._create('/os-aggregates/%s/action' % base.getid(aggregate), body, 'aggregate')

    def remove_host(self, aggregate, host):
        """Remove a host from the Host Aggregate."""
        body = {'remove_host': {'host': host}}
        return self._create('/os-aggregates/%s/action' % base.getid(aggregate), body, 'aggregate')

    def set_metadata(self, aggregate, metadata):
        """Set aggregate metadata, replacing the existing metadata."""
        body = {'set_metadata': {'metadata': metadata}}
        return self._create('/os-aggregates/%s/action' % base.getid(aggregate), body, 'aggregate')

    def delete(self, aggregate):
        """
        Delete the specified aggregate.

        :param aggregate: The aggregate to delete
        :returns: An instance of novaclient.base.TupleWithMeta
        """
        return self._delete('/os-aggregates/%s' % base.getid(aggregate))

    @api_versions.wraps('2.81')
    def cache_images(self, aggregate, images):
        """
        Request images be cached on a given aggregate.

        :param aggregate: The aggregate to target
        :param images: A list of image IDs to request caching
        :returns: An instance of novaclient.base.TupleWithMeta
        """
        body = {'cache': [{'id': base.getid(image)} for image in images]}
        resp, body = self.api.client.post('/os-aggregates/%s/images' % base.getid(aggregate), body=body)
        return self.convert_into_with_meta(body, resp)