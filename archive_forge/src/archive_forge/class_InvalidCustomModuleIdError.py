from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidCustomModuleIdError(Error):
    """An error representing a custom module ID that does not conform to _CUSTOM_MODULE_ID_REGEX."""

    def __init__(self, bad_module_id_arg: str):
        if bad_module_id_arg is None:
            super(Error, self).__init__('Missing custom module ID.')
        else:
            super(Error, self).__init__(f'"{bad_module_id_arg}" is not a valid custom module ID. The ID should consist only of numbers and be 1-20 characters in length.')