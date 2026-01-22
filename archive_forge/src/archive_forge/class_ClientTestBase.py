import os
import time
import os_client_config
from oslo_utils import uuidutils
from tempest.lib.cli import base
from tempest.lib import exceptions
class ClientTestBase(base.ClientTestBase):
    """Base class for aodhclient tests.

    Establishes the aodhclient and retrieves the essential environment
    information.
    """

    def _get_clients(self):
        return AodhClient()

    def retry_aodh(self, retry, *args, **kwargs):
        result = ''
        while not result.strip() and retry > 0:
            result = self.aodh(*args, **kwargs)
            if not result:
                time.sleep(1)
                retry -= 1
        return result

    def aodh(self, *args, **kwargs):
        return self.clients.aodh(*args, **kwargs)

    def get_token(self):
        cloud = os.environ.get('OS_ADMIN_CLOUD', 'devstack-admin')
        if cloud is not None and cloud != '':
            conf = os_client_config.OpenStackConfig()
            region_conf = conf.get_one_cloud(cloud=cloud)
            return region_conf.get_auth().get_token(region_conf.get_session())
        else:
            return ''

    def details_multiple(self, output_lines, with_label=False):
        """Return list of dicts with item details from cli output tables.

        If with_label is True, key '__label' is added to each items dict.
        For more about 'label' see OutputParser.tables().

        NOTE(sileht): come from tempest-lib just because cliff use
        Field instead of Property as first columun header.
        """
        items = []
        tables_ = self.parser.tables(output_lines)
        for table_ in tables_:
            if 'Field' not in table_['headers'] or 'Value' not in table_['headers']:
                raise exceptions.InvalidStructure()
            item = {}
            for value in table_['values']:
                item[value[0]] = value[1]
            if with_label:
                item['__label'] = table_['label']
            items.append(item)
        return items