from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
class EnabledMetric(object):

    def __init__(self, connection=None, metric=None, granularity=None):
        self.connection = connection
        self.metric = metric
        self.granularity = granularity

    def __repr__(self):
        return 'EnabledMetric(%s, %s)' % (self.metric, self.granularity)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'Granularity':
            self.granularity = value
        elif name == 'Metric':
            self.metric = value