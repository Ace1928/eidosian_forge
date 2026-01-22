from oslo_config import cfg
from oslo_utils import importutils
from wsme.rest import json
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.common import crypt
from glance.common import exception
from glance.common import utils as common_utils
import glance.domain
import glance.domain.proxy
from glance.i18n import _
class ImageRepo(object):

    def __init__(self, context, db_api):
        self.context = context
        self.db_api = db_api

    def get(self, image_id):
        try:
            db_api_image = dict(self.db_api.image_get(self.context, image_id))
            if db_api_image['deleted']:
                raise exception.ImageNotFound()
        except (exception.ImageNotFound, exception.Forbidden):
            msg = _('No image found with ID %s') % image_id
            raise exception.ImageNotFound(msg)
        tags = self.db_api.image_tag_get_all(self.context, image_id)
        image = self._format_image_from_db(db_api_image, tags)
        return ImageProxy(image, self.context, self.db_api)

    def list(self, marker=None, limit=None, sort_key=None, sort_dir=None, filters=None, member_status='accepted'):
        sort_key = ['created_at'] if not sort_key else sort_key
        sort_dir = ['desc'] if not sort_dir else sort_dir
        db_api_images = self.db_api.image_get_all(self.context, filters=filters, marker=marker, limit=limit, sort_key=sort_key, sort_dir=sort_dir, member_status=member_status, return_tag=True)
        images = []
        for db_api_image in db_api_images:
            db_image = dict(db_api_image)
            image = self._format_image_from_db(db_image, db_image['tags'])
            images.append(image)
        return images

    def _format_image_from_db(self, db_image, db_tags):
        properties = {}
        for prop in db_image.pop('properties'):
            if not prop['deleted']:
                properties[prop['name']] = prop['value']
        locations = [loc for loc in db_image['locations'] if loc['status'] == 'active']
        if CONF.metadata_encryption_key:
            key = CONF.metadata_encryption_key
            for location in locations:
                location['url'] = crypt.urlsafe_decrypt(key, location['url'])
        if db_image['visibility'] == 'shared' and self.context.owner != db_image['owner']:
            member = self.context.owner
        else:
            member = None
        return glance.domain.Image(image_id=db_image['id'], name=db_image['name'], status=db_image['status'], created_at=db_image['created_at'], updated_at=db_image['updated_at'], visibility=db_image['visibility'], min_disk=db_image['min_disk'], min_ram=db_image['min_ram'], protected=db_image['protected'], locations=common_utils.sort_image_locations(locations), checksum=db_image['checksum'], os_hash_algo=db_image['os_hash_algo'], os_hash_value=db_image['os_hash_value'], owner=db_image['owner'], disk_format=db_image['disk_format'], container_format=db_image['container_format'], size=db_image['size'], virtual_size=db_image['virtual_size'], extra_properties=properties, tags=db_tags, os_hidden=db_image['os_hidden'], member=member)

    def _format_image_to_db(self, image):
        locations = image.locations
        if CONF.metadata_encryption_key:
            key = CONF.metadata_encryption_key
            ld = []
            for loc in locations:
                url = crypt.urlsafe_encrypt(key, loc['url'])
                ld.append({'url': url, 'metadata': loc['metadata'], 'status': loc['status'], 'id': loc.get('id')})
            locations = ld
        return {'id': image.image_id, 'name': image.name, 'status': image.status, 'created_at': image.created_at, 'min_disk': image.min_disk, 'min_ram': image.min_ram, 'protected': image.protected, 'locations': locations, 'checksum': image.checksum, 'os_hash_algo': image.os_hash_algo, 'os_hash_value': image.os_hash_value, 'owner': image.owner, 'disk_format': image.disk_format, 'container_format': image.container_format, 'size': image.size, 'virtual_size': image.virtual_size, 'visibility': image.visibility, 'properties': dict(image.extra_properties), 'os_hidden': image.os_hidden}

    def add(self, image):
        image_values = self._format_image_to_db(image)
        if image_values['size'] is not None and image_values['size'] > CONF.image_size_cap:
            raise exception.ImageSizeLimitExceeded
        image_values['updated_at'] = image.updated_at
        new_values = self.db_api.image_create(self.context, image_values)
        self.db_api.image_tag_set_all(self.context, image.image_id, image.tags)
        image.created_at = new_values['created_at']
        image.updated_at = new_values['updated_at']

    def save(self, image, from_state=None):
        image_values = self._format_image_to_db(image)
        if image_values['size'] is not None and image_values['size'] > CONF.image_size_cap:
            raise exception.ImageSizeLimitExceeded
        new_values = self.db_api.image_update(self.context, image.image_id, image_values, purge_props=True, from_state=from_state, atomic_props=IMAGE_ATOMIC_PROPS)
        self.db_api.image_tag_set_all(self.context, image.image_id, image.tags)
        image.updated_at = new_values['updated_at']

    def remove(self, image):
        try:
            self.db_api.image_update(self.context, image.image_id, {'status': image.status}, purge_props=True)
        except (exception.ImageNotFound, exception.Forbidden):
            msg = _('No image found with ID %s') % image.image_id
            raise exception.ImageNotFound(msg)
        new_values = self.db_api.image_destroy(self.context, image.image_id)
        image.updated_at = new_values['updated_at']

    def set_property_atomic(self, image, name, value):
        self.db_api.image_set_property_atomic(image.image_id, name, value)

    def delete_property_atomic(self, image, name, value):
        self.db_api.image_delete_property_atomic(image.image_id, name, value)