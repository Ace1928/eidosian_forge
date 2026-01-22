from boto.resultset import ResultSet
class JobFlow(EmrObject):
    Fields = set(['AmiVersion', 'AvailabilityZone', 'CreationDateTime', 'Ec2KeyName', 'EndDateTime', 'HadoopVersion', 'Id', 'InstanceCount', 'JobFlowId', 'KeepJobFlowAliveWhenNoSteps', 'LastStateChangeReason', 'LogUri', 'MasterInstanceId', 'MasterInstanceType', 'MasterPublicDnsName', 'Name', 'NormalizedInstanceHours', 'ReadyDateTime', 'RequestId', 'SlaveInstanceType', 'StartDateTime', 'State', 'TerminationProtected', 'Type', 'Value', 'VisibleToAllUsers'])

    def __init__(self, connection=None):
        self.connection = connection
        self.steps = None
        self.instancegroups = None
        self.bootstrapactions = None

    def startElement(self, name, attrs, connection):
        if name == 'Steps':
            self.steps = ResultSet([('member', Step)])
            return self.steps
        elif name == 'InstanceGroups':
            self.instancegroups = ResultSet([('member', InstanceGroup)])
            return self.instancegroups
        elif name == 'BootstrapActions':
            self.bootstrapactions = ResultSet([('member', BootstrapAction)])
            return self.bootstrapactions
        elif name == 'SupportedProducts':
            self.supported_products = ResultSet([('member', SupportedProduct)])
            return self.supported_products
        else:
            return None