import collections
import ipaddress
from oslo_utils import uuidutils
import re
import string
from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient.v2 import share_instances
class ShareManager(base.MetadataCapableManager):
    """Manage :class:`Share` resources."""
    resource_class = Share
    resource_path = '/shares'

    def create(self, share_proto, size, snapshot_id=None, name=None, description=None, metadata=None, share_network=None, share_type=None, is_public=False, availability_zone=None, share_group_id=None, scheduler_hints=None, return_raw=False):
        """Create a share.

        :param share_proto: text - share protocol for new share available
            values are NFS, CIFS, CephFS, GlusterFS, HDFS and MAPRFS.
        :param size: int - size in GiB
        :param snapshot_id: text - ID of the snapshot
        :param name: text - name of new share
        :param description: text - description of a share
        :param metadata: dict - optional metadata to set on share creation
        :param share_network: either instance of ShareNetwork or text with ID
        :param share_type: either instance of ShareType or text with ID
        :param is_public: bool, whether to set share as public or not.
        :param share_group_id: text - ID of the share group to which the share
            should belong
        :param scheduler_hints: dict - hints for the scheduler to place share
            on most appropriate host e.g. keys are same_host for affinity and
            different_host for anti-affinity
        :rtype: :class:`Share`
        """
        share_metadata = metadata if metadata is not None else dict()
        scheduler_hints = scheduler_hints if scheduler_hints is not None else dict()
        body = {'size': size, 'snapshot_id': snapshot_id, 'name': name, 'description': description, 'metadata': share_metadata, 'share_proto': share_proto, 'share_network_id': base.getid(share_network), 'share_type': base.getid(share_type), 'is_public': is_public, 'availability_zone': availability_zone, 'scheduler_hints': scheduler_hints}
        if share_group_id:
            body['share_group_id'] = share_group_id
        return self._create('/shares', {'share': body}, 'share', return_raw=return_raw)

    @api_versions.wraps('2.29')
    @api_versions.experimental_api
    def migration_start(self, share, host, force_host_assisted_migration, preserve_metadata, writable, nondisruptive, preserve_snapshots, new_share_network_id=None, new_share_type_id=None):
        return self._action('migration_start', share, {'host': host, 'force_host_assisted_migration': force_host_assisted_migration, 'preserve_metadata': preserve_metadata, 'preserve_snapshots': preserve_snapshots, 'writable': writable, 'nondisruptive': nondisruptive, 'new_share_network_id': new_share_network_id, 'new_share_type_id': new_share_type_id})

    @api_versions.wraps('2.22')
    @api_versions.experimental_api
    def reset_task_state(self, share, task_state):
        """Update the provided share with the provided task state.

        :param share: either share object or text with its ID.
        :param task_state: text with new task state to set for share.
        """
        return self._action('reset_task_state', share, {'task_state': task_state})

    @api_versions.wraps('2.22')
    @api_versions.experimental_api
    def migration_complete(self, share):
        """Completes migration for a given share.

        :param share: The :class:'share' to complete migration
        """
        return self._action('migration_complete', share)

    @api_versions.wraps('2.22')
    @api_versions.experimental_api
    def migration_cancel(self, share):
        """Attempts to cancel migration for a given share.

        :param share: The :class:'share' to cancel migration
        """
        return self._action('migration_cancel', share)

    @api_versions.wraps('2.22')
    @api_versions.experimental_api
    def migration_get_progress(self, share):
        """Obtains progress of share migration for a given share.

        :param share: The :class:'share' to obtain migration progress
        """
        return self._action('migration_get_progress', share)

    def _do_manage(self, service_host, protocol, export_path, driver_options=None, share_type=None, name=None, description=None, is_public=None, share_server_id=None, resource_path='/shares/manage'):
        """Manage some existing share.

        :param service_host: text - host where manila share service is running
        :param protocol: text - share protocol that is used
        :param export_path: text - export path of share
        :param driver_options: dict - custom set of key-values
        :param share_type: text - share type that should be used for share
        :param name: text - name of new share
        :param description: - description for new share
        :param is_public: - visibility for new share
        :param share_server_id: text - id of share server associated with share
        """
        driver_options = driver_options if driver_options else dict()
        body = {'service_host': service_host, 'share_type': share_type, 'protocol': protocol, 'export_path': export_path, 'driver_options': driver_options, 'name': name, 'description': description, 'share_server_id': share_server_id}
        if is_public is not None:
            body['is_public'] = is_public
        return self._create(resource_path, {'share': body}, 'share')

    @api_versions.wraps('1.0', '2.6')
    def manage(self, service_host, protocol, export_path, driver_options=None, share_type=None, name=None, description=None):
        return self._do_manage(service_host, protocol, export_path, driver_options=driver_options, share_type=share_type, name=name, description=description, resource_path='/os-share-manage')

    @api_versions.wraps('2.7', '2.7')
    def manage(self, service_host, protocol, export_path, driver_options=None, share_type=None, name=None, description=None):
        return self._do_manage(service_host, protocol, export_path, driver_options=driver_options, share_type=share_type, name=name, description=description, resource_path='/shares/manage')

    @api_versions.wraps('2.8', '2.48')
    def manage(self, service_host, protocol, export_path, driver_options=None, share_type=None, name=None, description=None, is_public=False):
        return self._do_manage(service_host, protocol, export_path, driver_options=driver_options, share_type=share_type, name=name, description=description, is_public=is_public, resource_path='/shares/manage')

    @api_versions.wraps('2.49')
    def manage(self, service_host, protocol, export_path, driver_options=None, share_type=None, name=None, description=None, is_public=False, share_server_id=None):
        return self._do_manage(service_host, protocol, export_path, driver_options=driver_options, share_type=share_type, name=name, description=description, is_public=is_public, share_server_id=share_server_id, resource_path='/shares/manage')

    @api_versions.wraps('1.0', '2.6')
    def unmanage(self, share):
        """Unmanage a share.

        :param share: either share object or text with its ID.
        """
        return self.api.client.post('/os-share-unmanage/%s/unmanage' % base.getid(share))

    @api_versions.wraps('2.7')
    def unmanage(self, share):
        """Unmanage a share.

        :param share: either share object or text with its ID.
        """
        return self._action('unmanage', share)

    @api_versions.wraps('2.27')
    def revert_to_snapshot(self, share, snapshot):
        """Reverts a share (in place) to a snapshot.

        The snapshot must be the most recent one known to manila.
        :param share: either share object or text with its ID.
        :param snapshot: either snapshot object or text with its ID.
        """
        snapshot_id = base.getid(snapshot)
        info = {'snapshot_id': snapshot_id}
        return self._action('revert', share, info=info)

    def get(self, share, return_raw=False):
        """Get a share.

        :param share: either share object or text with its ID.
        :rtype: :class:`Share`
        """
        share_id = base.getid(share)
        return self._get('/shares/%s' % share_id, 'share', return_raw=return_raw)

    def update(self, share, **kwargs):
        """Updates a share.

        :param share: either share object or text with its ID.
        :rtype: :class:`Share`
        """
        if not kwargs:
            return
        body = {'share': kwargs}
        share_id = base.getid(share)
        return self._update('/shares/%s' % share_id, body)

    @api_versions.wraps('1.0', '2.34')
    def list(self, detailed=True, search_opts=None, sort_key=None, sort_dir=None, return_raw=False):
        """Get a list of all shares."""
        search_opts = search_opts or {}
        search_opts.pop('export_location', None)
        search_opts.pop('is_soft_deleted', None)
        return self.do_list(detailed=detailed, search_opts=search_opts, sort_key=sort_key, sort_dir=sort_dir, return_raw=return_raw)

    @api_versions.wraps('2.35', '2.68')
    def list(self, detailed=True, search_opts=None, sort_key=None, sort_dir=None, return_raw=False):
        """Get a list of all shares."""
        if search_opts is None:
            search_opts = {}
        search_opts.pop('is_soft_deleted', None)
        return self.do_list(detailed=detailed, search_opts=search_opts, sort_key=sort_key, sort_dir=sort_dir, return_raw=return_raw)

    @api_versions.wraps('2.69')
    def list(self, detailed=True, search_opts=None, sort_key=None, sort_dir=None, return_raw=False):
        """Get a list of all shares."""
        return self.do_list(detailed=detailed, search_opts=search_opts, sort_key=sort_key, sort_dir=sort_dir, return_raw=return_raw)

    def do_list(self, detailed=True, search_opts=None, sort_key=None, sort_dir=None, return_raw=False):
        """Get a list of all shares.

        :param detailed: Whether to return detailed share info or not.
        :param search_opts: dict with search options to filter out shares.
            available keys are below (('name1', 'name2', ...), 'type'):
            - ('all_tenants', int)
            - ('is_public', bool)
            - ('metadata', dict)
            - ('extra_specs', dict)
            - ('limit', int)
            - ('offset', int)
            - ('name', text)
            - ('status', text)
            - ('host', text)
            - ('share_server_id', text)
            - (('share_network_id', 'share_network'), text)
            - (('share_type_id', 'share_type'), text)
            - (('snapshot_id', 'snapshot'), text)
            - ('is_soft_deleted', bool)
            Note, that member context will have restricted set of
            available search opts. For admin context filtering also available
            by each share attr from its Model. So, this list is not full for
            admin context.
        :param sort_key: Key to be sorted (i.e. 'created_at' or 'status').
        :param sort_dir: Sort direction, should be 'desc' or 'asc'.
        :rtype: list of :class:`Share`
        """
        if search_opts is None:
            search_opts = {}
        if sort_key is not None:
            if sort_key in constants.SHARE_SORT_KEY_VALUES:
                search_opts['sort_key'] = sort_key
                if sort_key == 'share_type':
                    search_opts['sort_key'] = 'share_type_id'
                elif sort_key == 'snapshot':
                    search_opts['sort_key'] = 'snapshot_id'
                elif sort_key == 'share_network':
                    search_opts['sort_key'] = 'share_network_id'
                elif sort_key == 'availability_zone':
                    search_opts['sort_key'] = 'availability_zone_id'
            else:
                raise ValueError('sort_key must be one of the following: %s.' % ', '.join(constants.SHARE_SORT_KEY_VALUES))
        if sort_dir is not None:
            if sort_dir in constants.SORT_DIR_VALUES:
                search_opts['sort_dir'] = sort_dir
            else:
                raise ValueError('sort_dir must be one of the following: %s.' % ', '.join(constants.SORT_DIR_VALUES))
        if 'is_public' not in search_opts:
            search_opts['is_public'] = True
        export_location = search_opts.pop('export_location', None)
        if export_location:
            if uuidutils.is_uuid_like(export_location):
                search_opts['export_location_id'] = export_location
            else:
                search_opts['export_location_path'] = export_location
        query_string = self._build_query_string(search_opts)
        if detailed:
            path = '/shares/detail%s' % (query_string,)
        else:
            path = '/shares%s' % (query_string,)
        return self._list(path, 'shares', return_raw=return_raw)

    def delete(self, share, share_group_id=None):
        """Delete a share.

        :param share: either share object or text with its ID.
        :param share_group_id: text - ID of the share group to which the share
            belongs
        """
        url = '/shares/%s' % base.getid(share)
        if share_group_id:
            url += '?share_group_id=%s' % share_group_id
        self._delete(url)

    def _do_force_delete(self, share, action_name):
        """Delete a share forcibly - share status will be avoided.

        :param share: either share object or text with its ID.
        """
        return self._action(action_name, share)

    @api_versions.wraps('1.0', '2.6')
    def force_delete(self, share):
        return self._do_force_delete(share, 'os-force_delete')

    @api_versions.wraps('2.7')
    def force_delete(self, share):
        return self._do_force_delete(share, 'force_delete')

    @api_versions.wraps('2.69')
    def soft_delete(self, share):
        """Soft delete a share - share will go to recycle bin.

        :param share: either share object or text with its ID.
        """
        return self._action('soft_delete', share)

    @api_versions.wraps('2.69')
    def restore(self, share):
        """Restore a share - share will restore from recycle bin.

        :param share: either share object or text with its ID.
        """
        return self._action('restore', share)

    @staticmethod
    def _validate_common_name(access):
        if len(access) == 0 or len(access) > 64:
            exc_str = 'Invalid CN (common name). Must be 1-64 chars long.'
            raise exceptions.CommandError(exc_str)
    '\n    for the reference specification for AD usernames, reference below links:\n\n    1:https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/\n            windows-server-2008-R2-and-2008/cc733146(v=ws.11)\n    2:https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/\n            windows-server-2000/bb726984(v=technet.10)\n    '

    @staticmethod
    def _validate_username(access):
        sole_periods_spaces_re = '[\\s|\\.]+$'
        valid_username_re = '.[^\\"\\/\\\\\\[\\]\\:\\;\\|\\=\\,\\+\\*\\?\\<\\>]{3,254}$'
        username = access
        if re.match(sole_periods_spaces_re, username):
            exc_str = 'Invalid user or group name,cannot consist solely of periods or spaces.'
            raise exceptions.CommandError(exc_str)
        if not re.match(valid_username_re, username):
            exc_str = 'Invalid user or group name. Must be 4-255 characters and consist of alphanumeric characters and exclude special characters "/\\[]:;|=,+*?<>'
            raise exceptions.CommandError(exc_str)

    @staticmethod
    def _validate_cephx_id(cephx_id):
        if not cephx_id:
            raise exceptions.CommandError('Ceph IDs may not be empty.')
        if not set(cephx_id) <= set(string.printable):
            raise exceptions.CommandError('Ceph IDs must consist of ASCII printable characters.')
        if '.' in cephx_id:
            raise exceptions.CommandError('Ceph IDs may not contain periods.')

    def _validate_access(self, access_type, access, valid_access_types=None, enable_ipv6=False):
        if not valid_access_types:
            valid_access_types = ('ip', 'user', 'cert')
        if access_type in valid_access_types:
            if access_type == 'ip':
                try:
                    if enable_ipv6:
                        ipaddress.ip_network(str(access))
                    else:
                        ipaddress.IPv4Network(str(access))
                except ValueError as error:
                    raise exceptions.CommandError(str(error))
            elif access_type == 'user':
                self._validate_username(access)
            elif access_type == 'cert':
                self._validate_common_name(access.strip())
            elif access_type == 'cephx':
                self._validate_cephx_id(access.strip())
        else:
            msg = 'Only following access types are supported: %s' % ', '.join(valid_access_types)
            raise exceptions.CommandError(msg)

    def _do_allow(self, share, access_type, access, access_level, action_name, metadata=None, lock_visibility=False, lock_deletion=False, lock_reason=None):
        """Allow access to a share.

        :param share: either share object or text with its ID.
        :param access_type: string that represents access type ('ip','domain')
        :param access: string that represents access ('127.0.0.1')
        :param access_level: string that represents access level ('rw', 'ro')
        :param metadata: A dict of key/value pairs to be set
        """
        access_params = {'access_type': access_type, 'access_to': access}
        if access_level:
            access_params['access_level'] = access_level
        if metadata:
            access_params['metadata'] = metadata
        if lock_visibility:
            access_params['lock_visibility'] = lock_visibility
        if lock_deletion:
            access_params['lock_deletion'] = lock_deletion
        if lock_reason:
            access_params['lock_reason'] = lock_reason
        access = self._action(action_name, share, access_params)[1]['access']
        return access

    @api_versions.wraps('1.0', '2.6')
    def allow(self, share, access_type, access, access_level, metadata=None):
        self._validate_access(access_type, access)
        return self._do_allow(share, access_type, access, access_level, 'os-allow_access')

    @api_versions.wraps('2.7', '2.12')
    def allow(self, share, access_type, access, access_level, metadata=None):
        self._validate_access(access_type, access)
        return self._do_allow(share, access_type, access, access_level, 'allow_access')

    @api_versions.wraps('2.13', '2.37')
    def allow(self, share, access_type, access, access_level, metadata=None):
        valid_access_types = ('ip', 'user', 'cert', 'cephx')
        self._validate_access(access_type, access, valid_access_types)
        return self._do_allow(share, access_type, access, access_level, 'allow_access')

    @api_versions.wraps('2.38', '2.44')
    def allow(self, share, access_type, access, access_level, metadata=None):
        valid_access_types = ('ip', 'user', 'cert', 'cephx')
        self._validate_access(access_type, access, valid_access_types, enable_ipv6=True)
        return self._do_allow(share, access_type, access, access_level, 'allow_access')

    @api_versions.wraps('2.45', '2.81')
    def allow(self, share, access_type, access, access_level, metadata=None):
        valid_access_types = ('ip', 'user', 'cert', 'cephx')
        self._validate_access(access_type, access, valid_access_types, enable_ipv6=True)
        return self._do_allow(share, access_type, access, access_level, 'allow_access', metadata=metadata)

    @api_versions.wraps('2.82')
    def allow(self, share, access_type, access, access_level, metadata=None, lock_visibility=False, lock_deletion=False, lock_reason=None):
        valid_access_types = ('ip', 'user', 'cert', 'cephx')
        self._validate_access(access_type, access, valid_access_types, enable_ipv6=True)
        return self._do_allow(share, access_type, access, access_level, 'allow_access', metadata=metadata, lock_visibility=lock_visibility, lock_deletion=lock_deletion, lock_reason=lock_reason)

    def _do_deny(self, share, access_id, action_name, unrestrict=False):
        """Deny access to a share.

        :param share: either share object or text with its ID.
        :param access_id: ID of share access rule
        """
        body = {'access_id': access_id}
        if unrestrict:
            body['unrestrict'] = True
        return self._action(action_name, share, body)

    @api_versions.wraps('1.0', '2.6')
    def deny(self, share, access_id):
        return self._do_deny(share, access_id, 'os-deny_access')

    @api_versions.wraps('2.7', '2.81')
    def deny(self, share, access_id):
        return self._do_deny(share, access_id, 'deny_access')

    @api_versions.wraps('2.82')
    def deny(self, share, access_id, unrestrict=False):
        return self._do_deny(share, access_id, 'deny_access', unrestrict=unrestrict)

    def _do_access_list(self, share, action_name):
        """Get access list to a share.

        :param share: either share object or text with its ID.
        """
        access_list = self._action(action_name, share)[1]['access_list']
        if access_list:
            t = collections.namedtuple('Access', list(access_list[0]))
            return [t(*value.values()) for value in access_list]
        else:
            return []

    @api_versions.wraps('1.0', '2.6')
    def access_list(self, share):
        return self._do_access_list(share, 'os-access_list')

    @api_versions.wraps('2.7', '2.44')
    def access_list(self, share):
        return self._do_access_list(share, 'access_list')

    def _action(self, action, share, info=None, **kwargs):
        """Perform a share 'action'.

        :param action: text with action name.
        :param share: either share object or text with its ID.
        :param info: dict with data for specified 'action'.
        :param kwargs: dict with data to be provided for action hooks.
        """
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/shares/%s/action' % base.getid(share)
        return self.api.client.post(url, body=body)

    def _do_reset_state(self, share, state, action_name):
        """Update the provided share with the provided state.

        :param share: either share object or text with its ID.
        :param state: text with new state to set for share.
        """
        return self._action(action_name, share, {'status': state})

    @api_versions.wraps('1.0', '2.6')
    def reset_state(self, share, state):
        return self._do_reset_state(share, state, 'os-reset_status')

    @api_versions.wraps('2.7')
    def reset_state(self, share, state):
        return self._do_reset_state(share, state, 'reset_status')

    def _do_extend(self, share, new_size, action_name, force=False):
        """Extend the size of the specified share.

        :param share: either share object or text with its ID.
        :param new_size: The desired size to extend share to.
        :param force: if set to True, the scheduler's capacity decisions are
                      not accounted for. Setting this parameter to True does
                      not mean that the request will always succeed.
        """
        req_body = {'new_size': new_size}
        if force:
            req_body['force'] = 'true'
        return self._action(action_name, share, req_body)

    @api_versions.wraps('1.0', '2.6')
    def extend(self, share, new_size):
        return self._do_extend(share, new_size, 'os-extend')

    @api_versions.wraps('2.7', '2.63')
    def extend(self, share, new_size):
        return self._do_extend(share, new_size, 'extend')

    @api_versions.wraps('2.64')
    def extend(self, share, new_size, force=False):
        return self._do_extend(share, new_size, 'extend', force=force)

    def _do_shrink(self, share, new_size, action_name):
        """Shrink the size of the specified share.

        :param share: either share object or text with its ID.
        :param new_size: The desired size to shrink share to.
        """
        return self._action(action_name, share, {'new_size': new_size})

    @api_versions.wraps('1.0', '2.6')
    def shrink(self, share, new_size):
        return self._do_shrink(share, new_size, 'os-shrink')

    @api_versions.wraps('2.7')
    def shrink(self, share, new_size):
        return self._do_shrink(share, new_size, 'shrink')

    def list_instances(self, share):
        """List instances of the specified share.

        :param share: either share object or text with its ID.
        """
        return self._list('/shares/%s/instances' % base.getid(share), 'share_instances', manager=share_instances.ShareInstanceManager(self))