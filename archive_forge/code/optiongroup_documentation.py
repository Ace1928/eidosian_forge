from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.resultset import ResultSet

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
    