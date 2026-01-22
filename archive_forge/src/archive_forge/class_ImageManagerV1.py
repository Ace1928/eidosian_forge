from saharaclient.api import base
class ImageManagerV1(_ImageManager):

    def update_tags(self, image_id, new_tags):
        """Update an Image tags.

        :param new_tags: list of tags that will replace currently
                              assigned  tags
        """
        old_image = self.get(image_id)
        old_tags = frozenset(old_image.tags)
        new_tags = frozenset(new_tags)
        to_add = list(new_tags - old_tags)
        to_remove = list(old_tags - new_tags)
        add_response, remove_response = (None, None)
        if to_add:
            add_response = self._post('/images/%s/tag' % image_id, {'tags': to_add}, 'image')
        if to_remove:
            remove_response = self._post('/images/%s/untag' % image_id, {'tags': to_remove}, 'image')
        return remove_response or add_response or self.get(image_id)