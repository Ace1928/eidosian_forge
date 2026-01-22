import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def _filter_images(images, filters, context, status='accepted', is_public=None, admin_as_user=False):
    filtered_images = []
    if 'properties' in filters:
        prop_filter = filters.pop('properties')
        filters.update(prop_filter)
    if status == 'all':
        status = None
    visibility = filters.pop('visibility', None)
    os_hidden = filters.pop('os_hidden', False)
    for image in images:
        member = image_member_find(context, image_id=image['id'], member=context.owner, status=status)
        is_member = len(member) > 0
        has_ownership = context.owner and image['owner'] == context.owner
        image_is_public = image['visibility'] == 'public'
        image_is_community = image['visibility'] == 'community'
        image_is_shared = image['visibility'] == 'shared'
        image_is_hidden = image['os_hidden'] == True
        acts_as_admin = context.is_admin and (not admin_as_user)
        can_see = image_is_public or image_is_community or has_ownership or (is_member and image_is_shared) or acts_as_admin
        if not can_see:
            continue
        if visibility:
            if visibility == 'public':
                if not image_is_public:
                    continue
            elif visibility == 'private':
                if not image['visibility'] == 'private':
                    continue
                if not (has_ownership or acts_as_admin):
                    continue
            elif visibility == 'shared':
                if not image_is_shared:
                    continue
            elif visibility == 'community':
                if not image_is_community:
                    continue
        elif not has_ownership and image_is_community:
            continue
        if is_public is not None:
            if not image_is_public == is_public:
                continue
        if os_hidden:
            if image_is_hidden:
                continue
        to_add = True
        for k, value in filters.items():
            key = k
            if k.endswith('_min') or k.endswith('_max'):
                key = key[0:-4]
                try:
                    value = int(value)
                except ValueError:
                    msg = _('Unable to filter on a range with a non-numeric value.')
                    raise exception.InvalidFilterRangeValue(msg)
            if k.endswith('_min'):
                to_add = image.get(key) >= value
            elif k.endswith('_max'):
                to_add = image.get(key) <= value
            elif k in ['created_at', 'updated_at']:
                attr_value = image.get(key)
                operator, isotime = utils.split_filter_op(value)
                parsed_time = timeutils.parse_isotime(isotime)
                threshold = timeutils.normalize_time(parsed_time)
                to_add = utils.evaluate_filter_op(attr_value, operator, threshold)
            elif k in ['name', 'id', 'status', 'container_format', 'disk_format']:
                attr_value = image.get(key)
                operator, list_value = utils.split_filter_op(value)
                if operator == 'in':
                    threshold = utils.split_filter_value_for_quotes(list_value)
                    to_add = attr_value in threshold
                elif operator == 'eq':
                    to_add = attr_value == list_value
                else:
                    msg = _("Unable to filter by unknown operator '%s'.") % operator
                    raise exception.InvalidFilterOperatorValue(msg)
            elif k != 'is_public' and image.get(k) is not None:
                to_add = image.get(key) == value
            elif k == 'tags':
                filter_tags = value
                image_tags = image_tag_get_all(context, image['id'])
                for tag in filter_tags:
                    if tag not in image_tags:
                        to_add = False
                        break
            else:
                to_add = False
                for p in image['properties']:
                    properties = {p['name']: p['value'], 'deleted': p['deleted']}
                    to_add |= properties.get(key) == value and properties.get('deleted') is False
            if not to_add:
                break
        if to_add:
            filtered_images.append(image)
    return filtered_images