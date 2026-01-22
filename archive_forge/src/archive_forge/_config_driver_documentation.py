from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
from castellan import key_manager
from oslo_config import cfg
from oslo_config import sources
from oslo_log import log
A configuration source for configuration values served through castellan.  # noqa

    :param config_file: The path to a castellan configuration file.

    :param mapping_file: The path to a configuration/castellan_id mapping file.
    