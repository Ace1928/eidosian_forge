from openstackclient.api import api
class APIv1(api.BaseAPI):
    """Image v1 API"""
    _endpoint_suffix = '/v1'

    def __init__(self, endpoint=None, **kwargs):
        super(APIv1, self).__init__(endpoint=endpoint, **kwargs)
        self.endpoint = self.endpoint.rstrip('/')
        self._munge_url()

    def _munge_url(self):
        if not self.endpoint.endswith(self._endpoint_suffix):
            self.endpoint = self.endpoint + self._endpoint_suffix

    def image_list(self, detailed=False, public=False, private=False, **filter):
        """Get available images

        :param detailed:
            Retrieve detailed response from server if True
        :param public:
            Return public images if True
        :param private:
            Return private images if True

        If public and private are both True or both False then all images are
        returned.  Both arguments False is equivalent to no filter and all
        images are returned.  Both arguments True is a filter that includes
        both public and private images which is the same set as all images.

        http://docs.openstack.org/api/openstack-image-service/1.1/content/requesting-a-list-of-public-vm-images.html
        http://docs.openstack.org/api/openstack-image-service/1.1/content/requesting-detailed-metadata-on-public-vm-images.html
        http://docs.openstack.org/api/openstack-image-service/1.1/content/filtering-images-returned-via-get-images-and-get-imagesdetail.html
        """
        url = '/images'
        if detailed or public or private:
            url += '/detail'
        image_list = self.list(url, **filter)['images']
        if public != private:
            image_list = [i for i in image_list if i['is_public'] == public]
        return image_list