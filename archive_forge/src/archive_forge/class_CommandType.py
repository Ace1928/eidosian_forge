from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
class CommandType(Enum):
    """An enum for the types of commands the generator supports."""
    DESCRIBE = 1
    LIST = 2
    DELETE = 3
    IMPORT = 4
    EXPORT = 5
    CONFIG_EXPORT = 6
    CREATE = 7
    WAIT = 8
    UPDATE = 9
    GET_IAM_POLICY = 10
    SET_IAM_POLICY = 11
    ADD_IAM_POLICY_BINDING = 12
    REMOVE_IAM_POLICY_BINDING = 13
    GENERIC = 14

    @property
    def default_method(self):
        """Returns the default API method name for this type of command."""
        return _DEFAULT_METHODS_BY_COMMAND_TYPE.get(self)

    @classmethod
    def ForName(cls, name):
        try:
            return CommandType[name.upper()]
        except KeyError:
            return CommandType.GENERIC

    @classmethod
    def HasRequestMethod(cls, name):
        methodless_commands = {cls.CONFIG_EXPORT}
        return name not in methodless_commands