from oslo_utils import encodeutils
from urllib import parse
from heatclient.common import base
from heatclient.common import utils
class ResourceTypeManager(base.BaseManager):
    resource_class = ResourceType
    KEY = 'resource_types'

    def list(self, **kwargs):
        """Get a list of resource types.

        :rtype: list of :class:`ResourceType`
        """
        url = '/%s' % self.KEY
        params = {}
        if 'filters' in kwargs:
            filters = kwargs.pop('filters')
            params.update(filters)
        if 'with_description' in kwargs:
            with_description = kwargs.pop('with_description')
            params.update({'with_description': with_description})
        if params:
            url += '?%s' % parse.urlencode(params, True)
        return self._list(url, self.KEY)

    def get(self, resource_type, with_description=False):
        """Get the details for a specific resource_type.

        :param resource_type: name of the resource type to get the details for
        :param with_description: return result with description or not
        """
        url_str = '/%s/%s' % (self.KEY, parse.quote(encodeutils.safe_encode(resource_type)))
        resp = self.client.get(url_str, params={'with_description': with_description})
        body = utils.get_response_body(resp)
        return body

    def generate_template(self, resource_type, template_type='cfn'):
        url_str = '/%s/%s/template' % (self.KEY, parse.quote(encodeutils.safe_encode(resource_type)))
        if template_type:
            url_str += '?%s' % parse.urlencode({'template_type': template_type}, True)
        resp = self.client.get(url_str)
        body = utils.get_response_body(resp)
        return body