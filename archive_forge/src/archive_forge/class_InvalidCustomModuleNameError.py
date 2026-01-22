from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidCustomModuleNameError(Error):
    """An error representing an invalid custom module name."""

    def __init__(self, bad_module_name_arg: str, module_type: str):
        valid_formats = '\n\n\t\t'.join(_GetValidNameFormatForModule(module_type))
        super(Error, self).__init__(f'"{bad_module_name_arg}" is not a valid custom module name.\n\n\tThe expected format is one of:\n\n\t\t{valid_formats}\n')