from oslo_utils import strutils
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import utils
class FlavorManager(base.ManagerWithFind):
    """Manage :class:`Flavor` resources."""
    resource_class = Flavor
    is_alphanum_id_allowed = True

    def list(self, detailed=True, is_public=True, marker=None, min_disk=None, min_ram=None, limit=None, sort_key=None, sort_dir=None):
        """Get a list of all flavors.

        :param detailed: Whether flavor needs to be return with details
                         (optional).
        :param is_public: Filter flavors with provided access type (optional).
                          None means give all flavors and only admin has query
                          access to all flavor types.
        :param marker: Begin returning flavors that appear later in the flavor
                       list than that represented by this flavor id (optional).
        :param min_disk: Filters the flavors by a minimum disk space, in GiB.
        :param min_ram: Filters the flavors by a minimum RAM, in MiB.
        :param limit: maximum number of flavors to return (optional).
                      Note the API server has a configurable default limit.
                      If no limit is specified here or limit is larger than
                      default, the default limit will be used.
        :param sort_key: Flavors list sort key (optional).
        :param sort_dir: Flavors list sort direction (optional).
        :returns: list of :class:`Flavor`.
        """
        qparams = {}
        if marker:
            qparams['marker'] = str(marker)
        if min_disk:
            qparams['minDisk'] = int(min_disk)
        if min_ram:
            qparams['minRam'] = int(min_ram)
        if limit:
            qparams['limit'] = int(limit)
        if sort_key:
            qparams['sort_key'] = str(sort_key)
        if sort_dir:
            qparams['sort_dir'] = str(sort_dir)
        if not is_public:
            qparams['is_public'] = is_public
        detail = ''
        if detailed:
            detail = '/detail'
        return self._list('/flavors%s' % detail, 'flavors', filters=qparams)

    def get(self, flavor):
        """Get a specific flavor.

        :param flavor: The ID of the :class:`Flavor` to get.
        :returns: :class:`Flavor`
        """
        return self._get('/flavors/%s' % base.getid(flavor), 'flavor')

    def delete(self, flavor):
        """Delete a specific flavor.

        :param flavor: Instance of :class:`Flavor` to delete or ID of the
                       flavor to delete.
        :returns: An instance of novaclient.base.TupleWithMeta
        """
        return self._delete('/flavors/%s' % base.getid(flavor))

    def _build_body(self, name, ram, vcpus, disk, id, swap, ephemeral, rxtx_factor, is_public):
        return {'flavor': {'name': name, 'ram': ram, 'vcpus': vcpus, 'disk': disk, 'id': id, 'swap': swap, 'OS-FLV-EXT-DATA:ephemeral': ephemeral, 'rxtx_factor': rxtx_factor, 'os-flavor-access:is_public': is_public}}

    def create(self, name, ram, vcpus, disk, flavorid='auto', ephemeral=0, swap=0, rxtx_factor=1.0, is_public=True, description=None):
        """Create a flavor.

        :param name: Descriptive name of the flavor
        :param ram: Memory in MiB for the flavor
        :param vcpus: Number of VCPUs for the flavor
        :param disk: Size of local disk in GiB
        :param flavorid: ID for the flavor (optional). You can use the reserved
                         value ``"auto"`` to have Nova generate a UUID for the
                         flavor in cases where you cannot simply pass ``None``.
        :param ephemeral: Ephemeral disk space in GiB.
        :param swap: Swap space in MiB
        :param rxtx_factor: RX/TX factor
        :param is_public: Whether or not the flavor is public.
        :param description: A free form description of the flavor.
                            Limited to 65535 characters in length.
                            Only printable characters are allowed.
                            (Available starting with microversion 2.55)
        :returns: :class:`Flavor`
        """
        try:
            ram = int(ram)
        except (TypeError, ValueError):
            raise exceptions.CommandError(_('Ram must be an integer.'))
        try:
            vcpus = int(vcpus)
        except (TypeError, ValueError):
            raise exceptions.CommandError(_('VCPUs must be an integer.'))
        try:
            disk = int(disk)
        except (TypeError, ValueError):
            raise exceptions.CommandError(_('Disk must be an integer.'))
        if flavorid == 'auto':
            flavorid = None
        try:
            swap = int(swap)
        except (TypeError, ValueError):
            raise exceptions.CommandError(_('Swap must be an integer.'))
        try:
            ephemeral = int(ephemeral)
        except (TypeError, ValueError):
            raise exceptions.CommandError(_('Ephemeral must be an integer.'))
        try:
            rxtx_factor = float(rxtx_factor)
        except (TypeError, ValueError):
            raise exceptions.CommandError(_('rxtx_factor must be a float.'))
        try:
            is_public = strutils.bool_from_string(is_public, True)
        except Exception:
            raise exceptions.CommandError(_('is_public must be a boolean.'))
        supports_description = api_versions.APIVersion('2.55')
        if description and self.api_version < supports_description:
            raise exceptions.UnsupportedAttribute('description', '2.55')
        body = self._build_body(name, ram, vcpus, disk, flavorid, swap, ephemeral, rxtx_factor, is_public)
        if description:
            body['flavor']['description'] = description
        return self._create('/flavors', body, 'flavor')

    @api_versions.wraps('2.55')
    def update(self, flavor, description=None):
        """
        Update the description of the flavor.

        :param flavor: The :class:`Flavor` (or its ID) to update.
        :param description: The description to set on the flavor.
        """
        body = {'flavor': {'description': description}}
        return self._update('/flavors/%s' % base.getid(flavor), body, 'flavor')