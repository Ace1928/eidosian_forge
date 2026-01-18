from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
def wait_table_exists(module, wait_timeout, table_name):
    _do_wait(module, 'table_exists', 'table creation', wait_timeout, table_name)