import json
import os
from os.path import isfile
from os.path import join
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy.schema import MetaData
from sqlalchemy.sql import select
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def _populate_metadata(meta, conn, metadata_path=None, merge=False, prefer_new=False, overwrite=False):
    if not metadata_path:
        metadata_path = CONF.metadata_source_path
    try:
        if isfile(metadata_path):
            json_schema_files = [metadata_path]
        else:
            json_schema_files = [f for f in os.listdir(metadata_path) if isfile(join(metadata_path, f)) and f.endswith('.json')]
    except OSError as e:
        LOG.error(encodeutils.exception_to_unicode(e))
        return
    if not json_schema_files:
        LOG.error(_LE('Json schema files not found in %s. Aborting.'), metadata_path)
        return
    namespaces_table = get_metadef_namespaces_table(meta, conn)
    namespace_rt_table = get_metadef_namespace_resource_types_table(meta, conn)
    objects_table = get_metadef_objects_table(meta, conn)
    tags_table = get_metadef_tags_table(meta, conn)
    properties_table = get_metadef_properties_table(meta, conn)
    resource_types_table = get_metadef_resource_types_table(meta, conn)
    for json_schema_file in json_schema_files:
        try:
            file = join(metadata_path, json_schema_file)
            with open(file) as json_file:
                metadata = json.load(json_file)
        except Exception as e:
            LOG.error(_LE('Failed to parse json file %(file_path)s while populating metadata due to: %(error_msg)s'), {'file_path': file, 'error_msg': encodeutils.exception_to_unicode(e)})
            continue
        values = {'namespace': metadata.get('namespace'), 'display_name': metadata.get('display_name'), 'description': metadata.get('description'), 'visibility': metadata.get('visibility'), 'protected': metadata.get('protected'), 'owner': metadata.get('owner', 'admin')}
        with conn.begin():
            db_namespace = conn.execute(select(namespaces_table.c.id).where(namespaces_table.c.namespace == values['namespace']).select_from(namespaces_table)).fetchone()
        if db_namespace and overwrite:
            LOG.info(_LI('Overwriting namespace %s'), values['namespace'])
            _clear_namespace_metadata(meta, db_namespace[0])
            db_namespace = None
        if not db_namespace:
            values.update({'created_at': timeutils.utcnow()})
            _insert_data_to_db(conn, namespaces_table, values)
            with conn.begin():
                db_namespace = conn.execute(select(namespaces_table.c.id).where(namespaces_table.c.namespace == values['namespace']).select_from(namespaces_table)).fetchone()
        elif not merge:
            LOG.info(_LI('Skipping namespace %s. It already exists in the database.'), values['namespace'])
            continue
        elif prefer_new:
            values.update({'updated_at': timeutils.utcnow()})
            _update_data_in_db(namespaces_table, values, namespaces_table.c.id, db_namespace[0])
        namespace_id = db_namespace[0]
        for resource_type in metadata.get('resource_type_associations', []):
            rt_id = _get_resource_type_id(meta, conn, resource_type['name'])
            if not rt_id:
                val = {'name': resource_type['name'], 'created_at': timeutils.utcnow(), 'protected': True}
                _insert_data_to_db(conn, resource_types_table, val)
                rt_id = _get_resource_type_id(meta, conn, resource_type['name'])
            elif prefer_new:
                val = {'updated_at': timeutils.utcnow()}
                _update_data_in_db(resource_types_table, val, resource_types_table.c.id, rt_id)
            values = {'namespace_id': namespace_id, 'resource_type_id': rt_id, 'properties_target': resource_type.get('properties_target'), 'prefix': resource_type.get('prefix')}
            namespace_resource_type = _get_namespace_resource_type_by_ids(meta, conn, namespace_id, rt_id)
            if not namespace_resource_type:
                values.update({'created_at': timeutils.utcnow()})
                _insert_data_to_db(conn, namespace_rt_table, values)
            elif prefer_new:
                values.update({'updated_at': timeutils.utcnow()})
                _update_rt_association(namespace_rt_table, values, rt_id, namespace_id)
        for name, schema in metadata.get('properties', {}).items():
            values = {'name': name, 'namespace_id': namespace_id, 'json_schema': json.dumps(schema)}
            property_id = _get_resource_id(properties_table, conn, namespace_id, name)
            if not property_id:
                values.update({'created_at': timeutils.utcnow()})
                _insert_data_to_db(conn, properties_table, values)
            elif prefer_new:
                values.update({'updated_at': timeutils.utcnow()})
                _update_data_in_db(properties_table, values, properties_table.c.id, property_id)
        for object in metadata.get('objects', []):
            values = {'name': object['name'], 'description': object.get('description'), 'namespace_id': namespace_id, 'json_schema': json.dumps(object.get('properties'))}
            object_id = _get_resource_id(objects_table, conn, namespace_id, object['name'])
            if not object_id:
                values.update({'created_at': timeutils.utcnow()})
                _insert_data_to_db(conn, objects_table, values)
            elif prefer_new:
                values.update({'updated_at': timeutils.utcnow()})
                _update_data_in_db(objects_table, values, objects_table.c.id, object_id)
        for tag in metadata.get('tags', []):
            values = {'name': tag.get('name'), 'namespace_id': namespace_id}
            tag_id = _get_resource_id(tags_table, conn, namespace_id, tag['name'])
            if not tag_id:
                values.update({'created_at': timeutils.utcnow()})
                _insert_data_to_db(conn, tags_table, values)
            elif prefer_new:
                values.update({'updated_at': timeutils.utcnow()})
                _update_data_in_db(tags_table, values, tags_table.c.id, tag_id)
        LOG.info(_LI('File %s loaded to database.'), file)
    LOG.info(_LI('Metadata loading finished'))