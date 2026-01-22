from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.resultset import ResultSet
class OptionGroupOption(object):
    """
    Describes a OptionGroupOption for use in an OptionGroup

    :ivar name: The name of the option
    :ivar description: The description of the option.
    :ivar engine_name: Engine name that this option can be applied to.
    :ivar major_engine_version: Indicates the major engine version that the
                                option is available for.
    :ivar min_minor_engine_version: The minimum required engine version for the
                                    option to be applied.
    :ivar permanent: Indicate if this option is permanent.
    :ivar persistent: Indicate if this option is persistent.
    :ivar port_required: Specifies whether the option requires a port.
    :ivar default_port: If the option requires a port, specifies the default
                        port for the option.
    :ivar settings: The option settings for this option.
    :ivar depends_on: List of all options that are prerequisites for this
                      option.
    """

    def __init__(self, name=None, description=None, engine_name=None, major_engine_version=None, min_minor_engine_version=None, permanent=False, persistent=False, port_required=False, default_port=None, settings=None, depends_on=None):
        self.name = name
        self.description = description
        self.engine_name = engine_name
        self.major_engine_version = major_engine_version
        self.min_minor_engine_version = min_minor_engine_version
        self.permanent = permanent
        self.persistent = persistent
        self.port_required = port_required
        self.default_port = default_port
        self.settings = settings
        self.depends_on = depends_on
        if self.settings is None:
            self.settings = []
        if self.depends_on is None:
            self.depends_on = []

    def __repr__(self):
        return 'OptionGroupOption:%s' % self.name

    def startElement(self, name, attrs, connection):
        if name == 'OptionGroupOptionSettings':
            self.settings = ResultSet([('OptionGroupOptionSettings', OptionGroupOptionSetting)])
        elif name == 'OptionsDependedOn':
            self.depends_on = []
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'Name':
            self.name = value
        elif name == 'Description':
            self.description = value
        elif name == 'EngineName':
            self.engine_name = value
        elif name == 'MajorEngineVersion':
            self.major_engine_version = value
        elif name == 'MinimumRequiredMinorEngineVersion':
            self.min_minor_engine_version = value
        elif name == 'Permanent':
            if value.lower() == 'true':
                self.permenant = True
            else:
                self.permenant = False
        elif name == 'Persistent':
            if value.lower() == 'true':
                self.persistent = True
            else:
                self.persistent = False
        elif name == 'PortRequired':
            if value.lower() == 'true':
                self.port_required = True
            else:
                self.port_required = False
        elif name == 'DefaultPort':
            self.default_port = int(value)
        else:
            setattr(self, name, value)