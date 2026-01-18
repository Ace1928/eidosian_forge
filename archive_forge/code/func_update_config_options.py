import abc
from keystone import exception
@abc.abstractmethod
def update_config_options(self, domain_id, option_list):
    """Update config options for a domain.

        :param domain_id: the domain for this option
        :param option_list: a list of dicts, each one specifying an option

        """
    raise exception.NotImplemented()