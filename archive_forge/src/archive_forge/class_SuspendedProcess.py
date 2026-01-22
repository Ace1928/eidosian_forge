from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
class SuspendedProcess(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.process_name = None
        self.reason = None

    def __repr__(self):
        return 'SuspendedProcess(%s, %s)' % (self.process_name, self.reason)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'ProcessName':
            self.process_name = value
        elif name == 'SuspensionReason':
            self.reason = value