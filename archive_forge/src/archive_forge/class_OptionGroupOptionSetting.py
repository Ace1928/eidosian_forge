from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.resultset import ResultSet
class OptionGroupOptionSetting(object):
    """
    Describes a OptionGroupOptionSetting for use in an OptionGroupOption.

    :ivar name: The name of the option that has settings that you can set.
    :ivar description: The description of the option setting.
    :ivar value: The current value of the option setting.
    :ivar default_value: The default value of the option setting.
    :ivar allowed_values: The allowed values of the option setting.
    :ivar data_type: The data type of the option setting.
    :ivar apply_type: The DB engine specific parameter type.
    :ivar is_modifiable: A Boolean value that, when true, indicates the option
                         setting can be modified from the default.
    :ivar is_collection: Indicates if the option setting is part of a
                         collection.
    """

    def __init__(self, name=None, description=None, default_value=False, allowed_values=None, apply_type=None, is_modifiable=False):
        self.name = name
        self.description = description
        self.default_value = default_value
        self.allowed_values = allowed_values
        self.apply_type = apply_type
        self.is_modifiable = is_modifiable

    def __repr__(self):
        return 'OptionGroupOptionSetting:%s' % self.name

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'SettingName':
            self.name = value
        elif name == 'SettingDescription':
            self.description = value
        elif name == 'DefaultValue':
            self.default_value = value
        elif name == 'AllowedValues':
            self.allowed_values = value
        elif name == 'ApplyType':
            self.apply_type = value
        elif name == 'IsModifiable':
            if value.lower() == 'true':
                self.is_modifiable = True
            else:
                self.is_modifiable = False
        else:
            setattr(self, name, value)