from openstackclient.api import api
Get available images

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
        